import torch.nn as nn
import torch
from VGG import VGGUnet, L2_norm, Encoder, Decoder
import data_utils
from jacobian import grid_sample
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformer import CrossAttention
from torchvision import transforms
import matplotlib.pyplot as plt
from visualize import *

to_pil_image = transforms.ToPILImage()


class Model(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.level = args.level
        self.SatFeatureNet = VGGUnet(self.level)
        self.GrdFeatureNet = VGGUnet(self.level)

        self.GrdEnc = Encoder()
        self.GrdDec = Decoder()

        self.HeightAttention = CrossAttention(dim=256, qkv_bias=False)
        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        
        torch.autograd.set_detect_anomaly(True)

    def sat2world(self, satmap_sidelength, pred_height):
        i = j =torch.arange(0, satmap_sidelength).cuda()
        ii, jj = torch.meshgrid(i, j)
        uv = torch.stack([jj, ii], dim=-1).float()
        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor([u0, v0]).cuda()

        meter_per_pixel = data_utils.get_meter_per_pixel()
        meter_per_pixel *= data_utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                    uv_center).unsqueeze(0).repeat(pred_height.shape[0], 1, 1, 1)  # shape = [satmap_sidelength, satmap_sidelength, 2]
        Y = -pred_height.permute(0, 2, 3, 1)
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :, :1], Y, XZ[:, :, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]
        return sat2realwap

    def World2GrdImgPixCoordinates(self, shift_u, shift_v, heading, camera_k, XYZ_1, grd_H, grd_W, ori_grdH, ori_grdW):
        B = heading.shape[0]

        shift_u_meters = self.args.shift_range_lon * shift_u
        shift_v_meters = self.args.shift_range_lat * shift_v
        heading = heading * self.args.rotation_range / 180 * torch.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = data_utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]

        # P = K[R|T]
        camera_k = camera_k.clone()
        camera_k[:, :1, :] = camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * grd_H / ori_grdH
        P = camera_k @ torch.cat([R, T], dim=-1)

        uv1 = torch.matmul(P[:, None, None, :, :], XYZ_1.unsqueeze(-1)).squeeze(-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        # mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        return uv
    
    def project_grd_to_map(self, grd_f, pred_height, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH, ori_grdW):
        '''
        grd_f: [B, C, H, W]
        pred_height: [B, 1, A, A]
        shift_u: [B, 1]
        shift_v: [B, 1]
        heading: [B, 1]
        camera_k: [B, 3, 3]
        satmap_sidelength: scalar
        ori_grdH: scalar
        ori_grdW: scalar
        '''

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength, pred_height)  # [ sidelength,sidelength,4]
        uv = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, camera_k, XYZ_1,
                                                   H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        # [B, H, W, 2], [2, B, H, W, 2], [1, B, H, W, 2]

        grd_f_trans, _ = grid_sample(grd_f, uv, jac=None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]

        return grd_f_trans

    def forward_project(self, image_tensor, camera_k, depth, meter_per_pixel, sat_width=512, ori_grdH=256, ori_grdW=1024):
        origin_image_tensor = image_tensor.clone()
        B, C, grd_H, grd_W = image_tensor.shape
        camera_k = camera_k.clone()
        camera_k[:, :1, :] = camera_k[:, :1,
                                :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * grd_H / ori_grdH
        # meter_per_pixel = 1
        image_tensor = image_tensor.permute(0,2,3,1).contiguous().view(B*grd_H*grd_W, -1)

        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                                torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to('cuda')
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]


        depth = depth.unsqueeze(-1)
        depth = F.interpolate(depth.permute(0,3,1,2), size=(grd_H, grd_W), mode='bilinear', align_corners=False).permute(0,2,3,1)
        # xyz_grd = xyz_w * depth / meter_per_pixel
        xyz_grd = xyz_w * depth * 1.2

        # xyz_grd = xyz_grd.long()
        # xyz_grd[:,:,:,0:1] += sat_width // 2
        # xyz_grd[:,:,:,2:3] += sat_width // 2
        # B, H, W, C = xyz_grd.shape
        xyz_grd = xyz_grd.view(B*grd_H*grd_W, -1)
        xyz_grd[:, 0] = xyz_grd[:, 0] / meter_per_pixel
        xyz_grd[:, 2] = xyz_grd[:, 2] / meter_per_pixel
        xyz_grd[:, 0] = xyz_grd[:, 0].long()
        xyz_grd[:, 2] = xyz_grd[:, 2].long()

        batch_ix = torch.cat([torch.full([grd_H*grd_W, 1], ix, device=image_tensor.device) for ix in range(B)], dim=0)
        xyz_grd = torch.cat([xyz_grd, batch_ix], dim=-1)

        kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

        xyz_grd_kept = xyz_grd[kept]
        image_tensor_kept = image_tensor[kept]

        max_height = xyz_grd_kept[:,1].max()

        xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
        xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
        xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
        xyz_grd_kept = xyz_grd_kept[:,[2,0,1,3]]
        rank = torch.stack((xyz_grd_kept[:, 0] * sat_width * B + (xyz_grd_kept[:, 1] + 1) * B + xyz_grd_kept[:, 3], xyz_grd_kept[:, 2]), dim=1)
        sorts_second = torch.argsort(rank[:, 1])
        xyz_grd_kept = xyz_grd_kept[sorts_second]
        image_tensor_kept = image_tensor_kept[sorts_second]
        sorted_rank = rank[sorts_second]
        sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
        xyz_grd_kept = xyz_grd_kept[sorts_first]
        image_tensor_kept = image_tensor_kept[sorts_first]
        sorted_rank = sorted_rank[sorts_first]
        kept = torch.ones_like(sorted_rank[:, 0])
        kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]
        res_xyz = xyz_grd_kept[kept.bool()]
        res_image = image_tensor_kept[kept.bool()]
        
        # grd_image_index = torch.cat((-res_xyz[:,1:2] + grd_image_width - 1,-res_xyz[:,0:1] + grd_image_height - 1), dim=-1)
        final = torch.zeros(B,sat_width,sat_width,C).to(torch.float32).to('cuda')
        sat_height = torch.zeros(B,sat_width,sat_width,1).to(torch.float32).to('cuda')
        final[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_image

        res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
        sat_height[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_xyz[:,2].unsqueeze(-1)
        sat_height = sat_height.permute(0,3,1,2)
        # img_num = 0
        # project_grd_img = to_pil_image(final[img_num].permute(2, 0, 1))
        # project_grd_img.save('sat_feat.png')

        # project_grd_img = to_pil_image(origin_image_tensor[img_num])
        # project_grd_img.save('grd_feat.png')

        return final.permute(0,3,1,2)

    def direct_map(self, sat_map, project_map, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            project_map: [B, C, H, W]
            mode:

        Returns:

        '''

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(project_map)

        # vis origin projection
        # grd_origin_proj = self.project_grd_to_map( grd_img_left, torch.zeros(B,1,512,512).to('cuda'), shift_u, shift_v, torch.zeros_like(heading), left_camera_k, 512, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(grd_origin_proj[0])
        # grd_project_img.save('grd_origin_proj.png')
        corr_maps = []

        # vis
        # img_num = 0
        # project_map_img = to_pil_image(project_map[img_num])
        # project_map_img.save('project_map.png')

        # sat_project_img = to_pil_image(sat_map[img_num])
        # sat_project_img.save('sat_map.png')
        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            # visulize feature map
            # sat_features_to_RGB(sat_feat, grd_feat)

            B, _, A, _ = sat_feat.shape

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]
    
    def feature_map(self, sat_map, grd_img_left, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train'):
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading

        # vis origin projection
        # meter_per_pixel = data_utils.get_meter_per_pixel()
        # grd_origin_proj = self.forward_project( grd_img_left, left_camera_k, grd_depth, meter_per_pixel, 512, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(grd_origin_proj[0])
        # grd_project_img.save('grd_origin_proj.png')
        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            # visulize feature map
            # sat_features_to_RGB(sat_feat, grd_feat_proj)

            A = sat_feat.shape[-1]
            B, C, H, W = grd_feat.shape

            grd_feat_proj = self.forward_project( grd_feat, left_camera_k, grd_depth, meter_per_pixel, A, ori_grdH, ori_grdW)

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]
    
    def feature_map_Unet(self, sat_map, grd_img_left, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train'):
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        sat8, sat4, sat2 = sat_feat_list

        grd8, grd4, grd2 = self.GrdEnc(grd_img_left)
        # [H/8, W/8] [H/4, W/4] [H/2, W/2]
        grd2sat8 = self.forward_project( grd8, left_camera_k, grd_depth, self.meters_per_pixel[0], sat8.shape[-1], ori_grdH, ori_grdW)
        grd2sat4 = self.forward_project( grd4, left_camera_k, grd_depth, self.meters_per_pixel[1], sat4.shape[-1], ori_grdH, ori_grdW)
        grd2sat2 = self.forward_project( grd2, left_camera_k, grd_depth, self.meters_per_pixel[2], sat2.shape[-1], ori_grdH, ori_grdW)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading

        grd_feat_list = self.GrdDec(grd2sat8, grd2sat4, grd2sat2)
        # vis origin projection
        # meter_per_pixel = data_utils.get_meter_per_pixel()
        # grd_origin_proj = self.forward_project( grd_img_left, left_camera_k, grd_depth, meter_per_pixel, 512, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(grd_origin_proj[0])
        # grd_project_img.save('grd_origin_proj.png')
        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            # visulize feature map
            # sat_features_to_RGB(sat_feat, grd_feat_proj)

            A = sat_feat.shape[-1]
            
            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)
        sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)

        gt_delta_x = - gt_shift_u[:, 0] * self.args.shift_range_lon
        gt_delta_y = - gt_shift_v[:, 0] * self.args.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))