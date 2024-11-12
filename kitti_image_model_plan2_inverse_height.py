import torch.nn as nn
import torch
from VGG import VGGUnet, L2_norm, Encoder, Decoder
from torch.nn.functional import normalize
import data_utils
from jacobian import grid_sample
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from visualize import *
from cross_attention import CrossViewAttention, BEVEmbedding

to_pil_image = transforms.ToPILImage()


class Model(nn.Module):
    def __init__(self, args, direct_map = False):  # device='cuda:0',
        super(Model, self).__init__()
        meter_per_pixel = data_utils.get_meter_per_pixel()

        self.args = args
        self.level = args.level
    
        self.SatFeatureNet = VGGUnet(self.level).eval()
        self.GrdFeatureNet = VGGUnet(self.level).eval()
        
        self.sat_embedding = nn.Parameter(torch.ones(1,256,64,64, device='cuda') * 0.5, requires_grad=True)
        # self.sat_embedding = nn.Embedding(100, 256)
        # self.w = nn.Parameter(torch.ones(1,256,1,1, device='cuda') * torch.log(torch.tensor(999.0)), requires_grad=True)
        self.w = nn.Parameter(torch.ones(1,256,64,64, device='cuda') * 0.5, requires_grad=True)
        self.proj = nn.Linear(256, 256)
        self.CVattn = nn.ModuleList()
        self.meters_per_pixel = []
        for level in range(1):
            self.CVattn.append(CrossViewAttention(blocks=1, dim=32 * (2 ** (3 - level)), qkv_bias=False).to('cuda'))
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        
        self.mse_loss = nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)

    def sat2world(self, satmap_sidelength, B, predict_height = None):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = data_utils.get_meter_per_pixel()
        meter_per_pixel *= data_utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center).repeat(B, 1, 1, 1)  # shape = [satmap_sidelength, satmap_sidelength, 2]
        if predict_height is not None:
            Y = -predict_height.permute(0, 2, 3, 1)
        else:
            Y = torch.zeros_like(XZ[:, :, :, :1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :, :1], Y, XZ[:, :, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W,
                                   ori_grdH, ori_grdW):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

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
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        P = camera_k @ torch.cat([R, T], dim=-1)

        uv1 = torch.matmul(P[:, None, None, :, :], XYZ_1.unsqueeze(-1)).squeeze(-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)

        return uv, mask
    
    def project_grd_to_map(self, grd_f, predict_height, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW):
        '''
        grd_f: [B, C, H, W]
        predict_height: [B, 1, H, W]
        shift_u: [B, 1]
        shift_v: [B, 1]
        heading: [B, 1]
        camera_k: [B, 3, 3]
        satmap_sidelength: scalar
        ori_grdH: scalar
        ori_grdW: scalar
        '''

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength, B, predict_height)  # [ sidelength,sidelength,4]
        uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k,
                                                   H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        # [B, H, W, 2], [2, B, H, W, 2], [1, B, H, W, 2]

        grd_f_trans, _ = grid_sample(grd_f, uv, jac=None)

        return grd_f_trans
    
    def project_sat_original(self, grd_f, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW):
        '''
        grd_f: [B, C, H, W]
        grd_c: [B, 1, H, W]
        shift_u: [B, 1]
        shift_v: [B, 1]
        heading: [B, 1]
        camera_k: [B, 3, 3]
        satmap_sidelength: scalar
        ori_grdH: scalar
        ori_grdW: scalar
        '''

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength, B)  # [ sidelength,sidelength,4]
        uv, mask1 = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k,
                                                   H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        grd_f_trans, mask2 = grid_sample(grd_f, uv, jac=None)
        mask = mask1.permute(0,3,1,2) & mask2
        return grd_f_trans, uv[..., 0], mask
    
    def predict_sat_height(self, sat_embedding, grd_feat, grd_height, u, left_camera_k, level, valid_index):
        predicted_sat_height = self.CVattn[level](sat_embedding, grd_feat, grd_height, u, left_camera_k, valid_index)
        return predicted_sat_height
    
    def inverse_map(self, sat_map, grd_img_left, left_camera_k, grd_height, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        with torch.no_grad():
            sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
            grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading
        corr_maps = []

        # self.CVattn(self.sat_random_feat[0], grd_feat_list[0], grd_height)
        # original vis    
        # test_proj, u, masks = self.project_sat_original(
        #         grd_img_left, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH, ori_grdW)
        # show_grd = test_proj[0,:,:,:]    
        # grd_left_image = to_pil_image(show_grd)
        # grd_left_image.save('grd_img_left.png')
        # show_sat = sat_map[0,:,:,:]    
        # sat_image = to_pil_image(show_sat)
        # sat_image.save('sat_image.png')
        # show_grd = test_proj[0,:,:,:]    
        # grd_image = to_pil_image(show_grd)
        # grd_image.save('grd_image.png')
        # show_grd = grd_img_left[0,:,:,:]    
        # grd_left_image = to_pil_image(show_grd)
        # grd_left_image.save('grd_img_left.png')
        # for i in range(test_proj.shape[1]):
        #     show_project = test_proj[0,i,0,:,:,:]
        #     # 将张量转换为NumPy数组，以便OpenCV处理
        #     image_np = show_project.permute(1, 2, 0).cpu().numpy() * 255
        #     image_np = image_np.astype(np.uint8).copy()
            
        #     # 将RGB通道转换为BGR通道
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #     cv2.circle(image_np, (int(shift_v[0,i,0] / meter_per_pixel + 256), int(-shift_u[0,i,0] / meter_per_pixel + 256)), radius=3, color=(0, 0, 255), thickness=-1)  # 红色圆点
        #     cv2.imwrite(f'pro_image{i}.png', image_np)

        for level in range(len(sat_feat_list)):
            if level != 0:
                continue
            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]
            B, C, H, W = grd_feat.shape
            A = sat_feat.shape[-1]

            feat_height = F.interpolate(grd_height.permute(0,3,1,2), size=(H, W), mode='bilinear', align_corners=True)
            
            grd_original_feat, u, masks = self.project_sat_original(
                grd_feat, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)
            
            valid_indexes = []
            # for i, mask in enumerate(masks.squeeze(1)):
            #     index = mask.nonzero()
            #     valid_indexes.append(index)
            # max_len = max([len(index) for index in valid_indexes])
            # sat_random_feat_rebatch = torch.zeros([B, C, max_len], dtype=torch.float32, requires_grad=False, device=sat_map.device)
            # u_rebatch = torch.zeros([B, max_len], dtype=torch.float32, requires_grad=False, device=sat_map.device)
            # for i in range(B):
            #     for j, index in enumerate(valid_indexes[i]):
            #         sat_random_feat_rebatch[i,:,j] = self.sat_random_feat[level][i,:,index[0],index[1]]
            #         u_rebatch[i,j] = u[i,index[0],index[1]]
            # sat_random_index = torch.zeros([B, A, A], dtype=torch.float32, requires_grad=False, device=sat_map.device)
            # sat_random_feat = self.sat_embedding(sat_random_index.long()).permute(0,3,1,2)
            predict_height = self.predict_sat_height(self.sat_embedding.repeat(B,1,1,1), grd_feat, feat_height, u, left_camera_k, level, valid_indexes)
            grd_height_feat = self.project_grd_to_map(
                grd_feat, predict_height, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            # vis height
            # 可视化第一个图像
            # see_num = 0
            # plt.imshow(predict_height.squeeze(1)[see_num].cpu().detach().numpy(), cmap='plasma')
            # plt.axis('off')  # 隐藏坐标轴
            # plt.colorbar()
            # plt.savefig('height.png')
            # plt.close()
            
            # grd_img_left_resize = F.interpolate(grd_img_left, size=(H, W), mode='bilinear', align_corners=True)
            # sat_img_resize = F.interpolate(sat_map, size=(A, A), mode='bilinear', align_corners=True)
            # test_height = self.project_grd_to_map(
            #     grd_img_left_resize, predict_height, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)
            # show_height = test_height[see_num,:,:,:]    
            # test_image = to_pil_image(show_height)
            # test_image.save('grd_height_image.png')

            # test_original, _, _ = self.project_sat_original(
            #     grd_img_left_resize, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)
            # show_original = test_original[see_num,:,:,:]    
            # test_image = to_pil_image(show_original)
            # test_image.save('grd_original_image.png')

            # show_grd = grd_img_left_resize[see_num,:,:,:]
            # test_image = to_pil_image(show_grd)
            # test_image.save('grd_img.png')

            # show_sat = sat_img_resize[see_num,:,:,:]
            # test_image = to_pil_image(show_sat)
            # test_image.save('sat_img.png')

            grd_feat_proj = grd_height_feat * self.sat_embedding + grd_original_feat * (1 - self.sat_embedding)

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

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

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