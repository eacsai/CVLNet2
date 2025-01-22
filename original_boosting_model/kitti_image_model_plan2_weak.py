import torch.nn as nn
import torch
from models.VGGW import VGGUnet, VGGUnet_G2S, Encoder, Decoder
import data_utils
from jacobian import grid_sample
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformer import CrossAttention
from torchvision import transforms
import matplotlib.pyplot as plt
from visualize import *

# from models.feature_extractor import FeatureExtractor
# from models.bev_net import BEVNet

to_pil_image = transforms.ToPILImage()


class Model(nn.Module):
    def __init__(self, args, device, direct_map = True):  # device='cuda:0',
        super(Model, self).__init__()
        self.device = device
        self.args = args
        self.level = args.level
        self.gs_channels = [32, 16, 4]
        self.SatFeatureNet = VGGUnet(self.level, self.gs_channels)
        
        self.GrdFeatureNet = VGGUnet(self.level, self.gs_channels)

        self.FeatureForT = VGGUnet(self.level, self.gs_channels)

        if not direct_map:
            self.ProjectFeatureNet = VGGUnet(self.level, self.gs_channels)

            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()

        self.HeightAttention = CrossAttention(dim=256, qkv_bias=False)
        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        
        # self.image_encoder = FeatureExtractor()
        # self.bev_net = BEVNet()
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

    def forward_project_v2(self, output_scores, grd_image, camera_k, satmap_sidelength, ori_grdH, ori_grdW):
        B, C, grd_H, grd_W = grd_image.shape
        camera_k = camera_k.clone()
        camera_k[:, :1, :] = camera_k[:, :1,
                                :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = camera_k[:, 1:2, :] * grd_H / ori_grdH
        
        weights = output_scores * torch.cumprod(torch.cat([torch.ones((output_scores.shape[0],1,output_scores.shape[2],output_scores.shape[3])).to('cuda'), (1. - output_scores)], 1), 1)[:,:-1,:,:]
        # depth_prob = torch.softmax(depth_scores, dim=1)
        # image_polar = torch.einsum("...dhw,...hwz->...dzw", image_tensor, depth_prob)
        image_polar = torch.einsum("...dhw,...hwz->...dzw", grd_image, weights)
        # grd_img_left_img = to_pil_image(image_polar[0])
        # grd_img_left_img.save('image_polar.png')

        f = camera_k[:, 0, 0][..., None, None]
        c = camera_k[:, 0, 2][..., None, None]

        z_max = 100
        depth_max = 80
        x_max = z_max / 2
        z_min = 0
        Δ = z_max / satmap_sidelength

        grid_xz = data_utils.make_grid(
            x_max * 2, z_max, step_y=Δ, step_x=Δ, orig_y=-50, orig_x=-x_max, y_up=False
        ).to('cuda')
        u = data_utils.from_homogeneous(grid_xz).squeeze(-1) * f + c
        # u= torch.flip(u, dims=[-1])
        z_idx = (grid_xz[..., 1] - z_min)
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)

        size = grid_polar.new_tensor([image_polar.shape[-1], depth_max])
        grid_uz_norm = (grid_polar * 2 / size) - 1
        # grid_uz_norm = grid_uz_norm * grid_polar.new_tensor([1, -1])  # y axis is up
        image_bev = F.grid_sample(image_polar, grid_uz_norm, align_corners=False)

        # visualize
        # origin_image_show = to_pil_image(grd_image[0])
        # origin_image_show.save('origin_image.png')
        # image_polar_show = to_pil_image(image_polar[0])
        # image_polar_show.save('image_polar.png')
        # image_bev_show = to_pil_image(image_bev[7])
        # image_bev_show.save('image_bev.png')
        return image_bev

    def direct_map(self, sat_map, grd_img, project_map, sat_Height, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            project_map: [B, C, H, W]
            mode:

        Returns:

        '''
        B = grd_img.shape[0]
        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(project_map)
        ori_grdH, ori_grdW = grd_img.shape[2], grd_img.shape[3]
        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading
        # vis origin projection
        img_num = 0
        grd_origin_proj = self.project_grd_to_map( grd_img, torch.zeros(B,1,512,512).to('cuda'), shift_u, shift_v, torch.zeros_like(heading), left_camera_k, 512, ori_grdH, ori_grdW)
        grd_project_img = to_pil_image(grd_origin_proj[img_num])
        grd_project_img.save('grd_origin_proj.png')
        corr_maps = []

        # vis
        project_map_img = to_pil_image(project_map[img_num])
        project_map_img.save('project_map.png')

        sat_project_img = to_pil_image(sat_map[img_num])
        sat_project_img.save('sat_map.png')

        grd_original_img = to_pil_image(grd_img[img_num])
        grd_original_img.save('grd_img.png')
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
    
    def feature_map(self, sat_map, grd_img_left, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None):
        level = 1
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.FeatureForT(sat_map)
        grd_feat_list, grd_conf_list = self.FeatureForT(grd_img_left)
        
        sat_feat = sat_feat_list[level]
        grd_feat = grd_feat_list[level]
        grd_conf = grd_conf_list[level]
        # grd_feat = self.image_encoder(grd_img_left)["feature_maps"][0]

        shift_lats = torch.zeros([B, 1, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_lons = torch.zeros([B, 1, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        thetas = torch.zeros([B, 1, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # self.forward_project_v2( grd_depth, grd_img_left, left_camera_k, 512, ori_grdH, ori_grdW)
        # vis origin projection
        # meter_per_pixel = data_utils.get_meter_per_pixel()
        # grd_origin_proj = self.forward_project_v2( grd_depth, grd_img_left, left_camera_k, 512, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(grd_origin_proj[0])
        # grd_project_img.save('grd_origin_proj.png')

        ideal_depth_values = torch.linspace(1,80,32).type(torch.float32)
        meter_per_pixel = self.meters_per_pixel[level]

        A = sat_feat.shape[-1]
        B, C, H, W = grd_feat.shape
        grd_feat_depth = F.interpolate(grd_depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        # height_map = grd_feat_depth[0]  # 现在形状为 [256, 1024]
        # plt.imshow(height_map.cpu().detach().numpy(), cmap='viridis')  # 使用 'viridis' 映射显示颜色
        # plt.colorbar(label='Satellite Height')
        # plt.title('Height Map Visualization')
        # plt.savefig('pred_height_img.png')
        # plt.close()
        # 理想的深度值列表
        # 计算每个元素到理想深度值的差异，得到误差矩阵
        # 使用 torch.abs(depth.unsqueeze(-1) - ideal_depth_values)
        error_matrix = torch.abs(grd_feat_depth.unsqueeze(-1) - ideal_depth_values.to(grd_depth.device))
        temperature = 1
        one_hot_depth = F.softmax(-error_matrix**2 / temperature, dim=-1)

        binary_tensor = (grd_feat_depth != 0).float()  # 生成一个二值化tensor，非零为1，零为0
        # 将tensor扩展到所需形状 [3, 3, 80]
        mask = binary_tensor.unsqueeze(-1).repeat(1, 1, 1, ideal_depth_values.shape[0])
        output_scores = one_hot_depth * mask

        # grd_image_sample = F.interpolate(grd_img_left, size=(H, W), mode='bilinear', align_corners=False)

        # res = self.forward_project_v2( output_scores, grd_image_sample, left_camera_k, A, ori_grdH, ori_grdW)

        # output_scores = output_scores.reshape(B, -1, 80)
        # output_scores = F.interpolate(output_scores, size=32, mode='linear', align_corners=True).reshape(8,32,128,512).permute(0, 2, 3, 1)
        grd_feat_proj = self.forward_project_v2( output_scores, grd_feat, left_camera_k, A, ori_grdH, ori_grdW)
        grd_conf_proj = self.forward_project_v2( output_scores, grd_conf, left_camera_k, A, ori_grdH, ori_grdW)
        # important_net
        # grd_feat_proj = self.bev_net(grd_feat_proj)["output"]
        
        # vis origin projection
        grd_image_sample = F.interpolate(grd_img_left, size=(H, W), mode='bilinear', align_corners=False)
        res = self.forward_project_v2( output_scores, grd_image_sample, left_camera_k, A, ori_grdH, ori_grdW)
        grd_project_img = to_pil_image(res[0])
        grd_project_img.save('grd_origin_proj.png')
        grd_img_left_img = to_pil_image(grd_img_left[0])
        grd_img_left_img.save('grd_img_left.png')
        sat_img = to_pil_image(sat_map[0])
        sat_img.save('sat_img.png')
        
        # visulize feature map
        sat_features_to_RGB(sat_feat, grd_feat_proj)
        
        crop_H = int(A - 20 * 3 / meter_per_pixel)
        crop_W = int(A - 20 * 3 / meter_per_pixel)
        g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
        # g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
        g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])

        sat_feat_dict_forT = {}
        sat_conf_dict_forT = {}
        g2s_feat_dict = {}
        g2s_conf_dict = {}

        g2s_feat_dict[level] = g2s_feat
        g2s_conf_dict[level] = g2s_conf
        render_loss = torch.tensor(0.0, device=g2s_feat.device)

        sat_feat_dict_forT[level] = sat_feat
        sat_conf_dict_forT[level] = sat_conf_list[level]

        return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, None, shift_lats, shift_lons, thetas, render_loss

    
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
            meter_per_pixel = self.meters_per_pixel[-2]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))
    
def batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, masks=None):
    '''
    compute corr_maps for training
    result corr_map has a shape of [M, N, H, W],
    M is the number of satellite images and N is the number of ground images
    '''

    levels = sorted([int(item) for item in args.level.split('_')])
    corr_maps = {}
    for _, level in enumerate(levels):
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape


        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)   # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(0, B):
                    denom_grd = torch.sum(F.conv2d(sat_conf_pow[i:i+1, :, :, :].repeat(1, C, 1, 1), g2s_feat_conf_pow), dim=1)
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

                # corr = corr / denominator_sat / denominator_grd

            else:

                # numerator
                signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_pow = (sat_feat).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

                # corr = corr / denominator_sat / denominator_grd

        else:
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()

            signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            # fixme: denominator
            # denominator_sat1 = []
            # mask_kernel = TF.center_crop(masks[level], [crop_H, crop_W]).float().unsqueeze(1).repeat(B, 1, 1, 1)
            # for i in range(0, B):
            #     denom_sat = torch.sum(F.conv2d(sat_feat.pow(2)[i, :, None, :, :], mask_kernel), dim=0)
            #     denominator_sat1.append(denom_sat)
            # denominator_sat1 = torch.sqrt(torch.stack(denominator_sat1, dim=0))  # [B (M), B (N), H, W]
            
            l2_norm_kernel = mask.repeat(1, C, 1, 1)
            sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)
            denominator_sat = torch.sqrt(sat_feat_squared_sum + 1e-8)
            # single_features_to_RGB(g2s_feat)
            # single_features_to_RGB(g2s_feat * mask)
            # original
            # denominator_sat_ori = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            # denominator_sat_ori = torch.sqrt(torch.sum(denominator_sat_ori, dim=1, keepdim=True))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = 2 - 2 * corr / denominator  # [B, B, H, W]

        corr_maps[level] = corr

    return corr_maps


def weak_supervise_loss(corr_maps):
    '''
    triplet loss/ metric learning loss for self-supervision
    corr_maps: dict
    key -- level; value -- corr map
    '''
    losses = []
    for key, corr in corr_maps:
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meter_per_pixels, GPS_error=5):
    '''
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []
    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
    # ------------------------------------------------

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

        # ---------- preparing for GPS error Loss -------
        meter_per_pixel = meter_per_pixels[level]
        w = (torch.round(W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)).long() # [B]
        h = (torch.round(H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)).long() # [B]
        radius = int(np.ceil(GPS_error / meter_per_pixel))
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius)
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius)
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius)
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius)
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))


def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meters_per_pixel):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    # cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    # sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    #
    # gt_delta_x = gt_shift_u[:, 0] * args.shift_range_lon
    # gt_delta_y = gt_shift_v[:, 0] * args.shift_range_lat
    #
    # gt_delta_x_rot = - gt_delta_x * cos - gt_delta_y * sin
    # gt_delta_y_rot = gt_delta_x * sin - gt_delta_y * cos

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        meter_per_pixel = meters_per_pixel[level]

        w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
        h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, meter_per_pixels, gt_heading, masks=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])
    meter_per_pixel = meter_per_pixels[level]

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

    if args.ConfGrd > 0:

        if args.ConfSat > 0:

            # numerator
            signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
            denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
            denominator_grd = torch.sqrt(denominator_grd)

        else:

            # numerator
            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

            # corr = corr / denominator_sat / denominator_grd

    else:

        signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        kernel = g2s_feat
        corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

        mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
        l2_norm_kernel = mask.repeat(1, C, 1, 1)
        sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)[0]
        denominator_sat = torch.maximum(torch.sqrt(sat_feat_squared_sum + 1e-8), torch.ones_like(sat_feat_squared_sum) * 1e-6)  # 滑动窗口的 L2 范数
        # denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        # denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))
        
        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator  # [B, H, W]

    corr_H = int(args.shift_range_lat * 3 / meter_per_pixel)
    corr_W = int(args.shift_range_lon * 3 / meter_per_pixel)

    corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)

    if args.visualize:
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * np.power(2, 3 - level)
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * np.power(2, 3 - level)
        return pred_u, pred_v, corr

    else:

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

        pred_u1 = pred_u * cos + pred_v * sin
        pred_v1 = - pred_u * sin + pred_v * cos

        return pred_u1, pred_v1, corr



def corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args,
                                              sat_uncer_dict=None):
    levels = [int(item) for item in args.level.split('_')]

    corr_maps = {}
    for level in levels:

        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape
        A = sat_feat.shape[2]

        # s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        # corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]
        #
        # if args.ConfGrd > 0:
        #     denominator = F.conv2d(sat_feat.pow(2).transpose(0, 1), g2s_conf.pow(2), groups=B).transpose(0, 1)
        # else:
        #     denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)

        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
                denominator_grd = torch.sqrt(denominator_grd)

            else:

                # numerator
                signal = sat_feat.reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        # if args.use_uncertainty:
        #     denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = corr / denominator

        corr_maps[level] = 2 - 2 * corr

    return corr_maps




def loss_func(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0, 0] - shift_lat_delta[-1, -1]  # scalar
    shift_lon_decrease = shift_lon_delta[0, 0] - shift_lon_delta[-1, -1]  # scalar
    thetas_decrease = thetas_delta[0, 0] - thetas_delta[-1, -1]  # scalar

    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
    loss_decrease = losses[0, 0] - losses[-1, -1]  # scalar
    loss = torch.mean(losses)  # mean or sum
    loss_last = losses[-1]

    return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
        shift_lat_delta[-1, -1], shift_lon_delta[-1, -1], thetas_delta[-1, -1]