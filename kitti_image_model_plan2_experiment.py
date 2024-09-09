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

from models.feature_extractor import FeatureExtractor
from models.bev_net import BEVNet
to_pil_image = transforms.ToPILImage()

class DepthPrediction(nn.Module):
    def __init__(self):
        super(DepthPrediction, self).__init__()
        # 定义一个简单的卷积层
        self.conv1 = nn.Conv2d(80 + 128, 80, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(80)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, bias=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, identity):        
        # 卷积 -> 批归一化 -> ReLU
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        
        # 跳跃连接：将输入 identity 加到输出
        res = out + identity
        res = self.conv2(res)
        # 应用 softmax 到输出结果
        # 这里假设你希望在 HxW 维度上应用 softmax
        res = F.softmax(res, dim=1)  # 通常在通道维度应用 softmax
        # res = self.sigmoid(res)
        return res

class Model(nn.Module):
    def __init__(self, args, direct_map = False):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.level = args.level
        self.SatFeatureNet = VGGUnet(self.level)
        self.depth_prediction = DepthPrediction()
        self.GrdFeatureNet = VGGUnet(self.level)
        if not direct_map:
            self.ProjectFeatureNet = VGGUnet(self.level)

            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()

        self.HeightAttention = CrossAttention(dim=256, qkv_bias=False)
        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        
        self.image_encoder = FeatureExtractor()
        self.bev_net = BEVNet()
        self.mse_loss = torch.nn.MSELoss()
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

        f = camera_k[:, 0, 0][..., None, None]
        c = camera_k[:, 0, 2][..., None, None]

        z_max = 100
        x_max = 50
        z_min = 0
        Δ = 100 / satmap_sidelength

        grid_xz = data_utils.make_grid(
            x_max * 2, z_max, step_y=Δ, step_x=Δ, orig_y=-50, orig_x=-x_max, y_up=False
        ).to('cuda')
        u = data_utils.from_homogeneous(grid_xz).squeeze(-1) * f + c
        # u= torch.flip(u, dims=[-1])
        z_idx = (grid_xz[..., 1] - z_min)
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)

        size = grid_polar.new_tensor(image_polar.shape[-2:][::-1])
        grid_uz_norm = (grid_polar * 2 / size) - 1
        # grid_uz_norm = grid_uz_norm * grid_polar.new_tensor([1, -1])  # y axis is up
        image_bev = F.grid_sample(image_polar, grid_uz_norm, align_corners=False)

        # visualize
        # origin_image_show = to_pil_image(grd_image[0])
        # origin_image_show.save('origin_image.png')
        # image_polar_show = to_pil_image(image_polar[0])
        # image_polar_show.save('image_polar.png')
        # image_bev_show = to_pil_image(image_bev[10])
        # image_bev_show.save('image_bev.png')
        return image_bev

    def create_depth_distribution(self, error_matrix, mu = 0, sigma = 1):
        normal_dist = torch.distributions.Normal(mu, sigma)
        scale = normal_dist.log_prob(torch.tensor([0]).to(error_matrix.device)).exp()
        return normal_dist.log_prob(error_matrix).exp() / scale  # 使用log_prob再取exp可以提高数值稳定性

    def direct_map(self, sat_map, grd_img, project_map, sat_Height, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            project_map: [B, C, H, W]
            mode:

        Returns:

        '''

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(project_map)

        # shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = gt_heading
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
    
    def feature_map(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train'):
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        sat_feat = sat_feat_list[-1]
        grd_feat = self.image_encoder(grd_img_left)["feature_maps"][0]

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading
        # self.forward_project_v2( grd_depth, grd_img_left, left_camera_k, 512, ori_grdH, ori_grdW)
        # vis origin projection
        # meter_per_pixel = data_utils.get_meter_per_pixel()
        # grd_origin_proj = self.forward_project( grd_img_left, left_camera_k, grd_depth, meter_per_pixel, 512, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(grd_origin_proj[0])
        # grd_project_img.save('grd_origin_proj.png')
        # grd_img_left_img = to_pil_image(grd_img_left[0])
        # grd_img_left_img.save('grd_img_left.png')

        corr_maps = []
        ideal_depth_values = torch.arange(80).type(torch.float32) + 1
        meter_per_pixel = self.meters_per_pixel[-2]

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

        gt_hot_depth = torch.nn.functional.one_hot(grd_feat_depth.long(), num_classes=81)
        gt_hot_depth = gt_hot_depth[..., 1:].float()

        error_matrix = torch.abs(grd_feat_depth.unsqueeze(-1) - ideal_depth_values.to(grd_depth.device))
        # temperature = 4
        # one_hot_depth = F.softmax(-error_matrix**2 / temperature, dim=-1)

        # 计算正态分布下每个x对应的概率密度值
        one_hot_depth = self.create_depth_distribution(error_matrix, sigma=2)  # 使用log_prob再取exp可以提高数值稳定性

        binary_tensor = (grd_feat_depth != 0).float()  # 生成一个二值化tensor，非零为1，零为0
        # 将tensor扩展到所需形状 [3, 3, 80]
        mask = binary_tensor.unsqueeze(-1).repeat(1, 1, 1, 80)

        # predict_depth = self.depth_prediction(torch.cat([grd_feat, one_hot_depth.permute(0,3,1,2)], dim=1), one_hot_depth.permute(0,3,1,2)).permute(0,2,3,1)
        # 将深度值矩阵与 one-hot 深度值矩阵相乘，得到深度值矩阵
        output_scores = one_hot_depth * mask

        # mse_loss
        # estimate_depth = output_scores * ideal_depth_values.to(output_scores.device)
        # estimate_depth = estimate_depth.sum(dim=-1, keepdim=False)
        # mse_loss = self.mse_loss(estimate_depth, grd_feat_depth)

        # grd_image_sample = F.interpolate(grd_img_left, size=(H, W), mode='bilinear', align_corners=False)
        grd_bev_proj = self.forward_project_v2( output_scores, grd_feat, left_camera_k, A, ori_grdH, ori_grdW)
        grd_bev_proj = self.bev_net(grd_bev_proj)
        grd_feat_proj = grd_bev_proj['output'] * grd_bev_proj.get('confidence').unsqueeze(1)
        # vis origin projection
        # grd_image_sample = F.interpolate(grd_img_left, size=(H, W), mode='bilinear', align_corners=False)
        # res = self.forward_project_v2( output_scores, grd_image_sample, left_camera_k, A, ori_grdH, ori_grdW)
        # grd_project_img = to_pil_image(res[1])
        # grd_project_img.save('grd_origin_proj.png')
        # visulize feature map
        # sat_features_to_RGB(sat_feat, grd_feat_proj)
        
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

    def inverse_map(self, sat_map, grd_img_left, project_map, sat_height, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)
        project_feat_list, project_conf_list = self.ProjectFeatureNet(project_map)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading

        g2s_corr_maps = []
        g2p_corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]
            project_feat = project_feat_list[level]

            # visulize feature map
            # sat_features_to_RGB(sat_feat, grd_feat)

            A = sat_feat.shape[-1]
            B, C, H, W = grd_feat.shape
            # if level == 0:
            #     sat_height = F.interpolate(sat_height.detach(), size=(A, A), mode='bilinear', align_corners=False)
            #     scale = C ** -0.5
            #     # pred_height =  self.HeightAttention(sat_feat_height, grd_feat_height, grd_height).view(B,1,A_max,A_max)
            #     sat_feat_reshaped = sat_feat.view(B, C, -1)  # (B, C, A*A)
            #     project_feat_reshaped = project_feat.view(B, C, -1)  # (B, C, A*A)
            #     sat_height_reshaped = sat_height.view(B, 1, -1)  # (B, 1, A*A)

            #     sat_feat_normalized = F.normalize(sat_feat_reshaped, p=2, dim=1, eps=1e-8)
            #     project_feat_normalized = F.normalize(project_feat_reshaped, p=2, dim=1, eps=1e-8)
            #     dot = torch.matmul(sat_feat_normalized.transpose(1, 2), project_feat_normalized) * scale
            #     temperature = 0.01
            #     dot = F.softmax(dot / temperature, dim=-1)
            #     pred_height = torch.matmul(dot, sat_height_reshaped.permute(0,2,1)).view(B, 1, A, A)
            # else:
            #     pred_height = F.interpolate(pred_height, size=(A, A), mode='bilinear', align_corners=False)

            sat_height = F.interpolate(sat_height.detach(), size=(A, A), mode='bilinear', align_corners=False)
            scale = C ** -0.5
            # pred_height =  self.HeightAttention(sat_feat_height, grd_feat_height, grd_height).view(B,1,A_max,A_max)
            sat_feat_reshaped = sat_feat.view(B, C, -1).permute(0,2,1)  # (B, A*A, C)
            project_feat_reshaped = project_feat.view(B, C, -1).permute(0,2,1)  # (B, A*A, C)
            sat_height_reshaped = sat_height.view(B, 1, -1).permute(0,2,1)  # (B, A*A, 1)

            pred_height = torch.tensor([]).to('cuda')
            for b in range(B):
                q = sat_feat_reshaped[b] # (A*A, C)
                k = project_feat_reshaped[b] # (A*A, C)
                v = sat_height_reshaped[b] # (A*A, 1)
                mask = v.clone()
                mask[mask != 0] = 1
                k = k[mask.squeeze(-1).bool()]
                v = v[mask.squeeze(-1).bool()]

                q_normalized = F.normalize(q, p=2, dim=1, eps=1e-8)
                k_normalized = F.normalize(k, p=2, dim=1, eps=1e-8)
                dot = torch.matmul(q_normalized, k_normalized.transpose(0, 1))
                temperature = 0.01
                dot = F.softmax(dot / temperature, dim=-1)
                res = torch.matmul(dot, v).view(A, A, 1).permute(2,0,1).unsqueeze(0) # (A*A, 1)
                pred_height = torch.cat((pred_height, res), dim=0)
            # visulize height map
            # 移除不必要的维度，得到形状为 [256, 1024]
            # img_num = 2
            # height_map = pred_height[img_num].squeeze(0)  # 现在形状为 [256, 1024]
            # plt.imshow(height_map.cpu().detach().numpy(), cmap='viridis')  # 使用 'viridis' 映射显示颜色
            # plt.colorbar(label='Satellite Height')
            # plt.title('Height Map Visualization')
            # plt.savefig('pred_height_img.png')
            # plt.close() 
            # sat_project_img = to_pil_image(sat_map[img_num])
            # sat_project_img.save('sat_map.png') 
            # plt.close()
            if not self.args.predict_height:
                pred_height = torch.zeros_like(pred_height)
            
            # vis projection
            # img_num = 2
            # pred_height = F.interpolate(pred_height, size=(512, 512), mode='bilinear', align_corners=False)
            # pred_height_project = self.project_grd_to_map( grd_img_left, pred_height, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH, ori_grdW)
            # grd_project_img = to_pil_image(pred_height_project[img_num])
            # grd_project_img.save('pred_height_project.png')

            # zero_height_project = self.project_grd_to_map( grd_img_left, -torch.zeros_like(pred_height), shift_u, shift_v, heading, left_camera_k, 512, ori_grdH, ori_grdW)
            # grd_project_img = to_pil_image(zero_height_project[img_num])
            # grd_project_img.save('zero_height_project.png')

            grd_feat_proj = self.project_grd_to_map( grd_feat, pred_height, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            g2p_feat = TF.center_crop(project_feat, [crop_H, crop_W])
            g2p_feat = F.normalize(g2p_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            
            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            
            # g2s_corr
            g2s_corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]
            g2s_corr = 2 - 2 * g2s_corr / denominator
            B, corr_H, corr_W = g2s_corr.shape
            g2s_corr_maps.append(g2s_corr)
            g2s_max_index = torch.argmin(g2s_corr.reshape(B, -1), dim=1)

            # g2p_corr
            g2p_corr = F.conv2d(s_feat, g2p_feat, groups=B)[0]  # [B, H, W]
            g2p_corr = 2 - 2 * g2p_corr / denominator
            B, corr_H, corr_W = g2p_corr.shape
            g2p_corr_maps.append(g2p_corr)
            g2p_max_index = torch.argmin(g2p_corr.reshape(B, -1), dim=1)

            max_index = (g2s_max_index + g2p_max_index) // 2
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * torch.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(g2s_corr_maps, gt_shift_u, gt_shift_v, gt_heading) + self.triplet_loss(g2p_corr_maps, gt_shift_u, gt_shift_v, gt_heading)
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