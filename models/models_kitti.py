import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import data_utils as utils
import os
import torchvision.transforms.functional as TF
from lpips import LPIPS
from loss.lpips import convert_to_buffer

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from models.VGGW import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec, FeatureHead
from jacobian import grid_sample

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
# from models_ford import loss_func
from models.swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1, TransOptimizerG2SP_V2
from models.cross_attention import CrossViewAttention
import copy

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from visualize import single_features_to_RGB, sat_features_to_RGB
from gaussian.encoder import GaussianEncoder
from gaussian.decoder import GrdDecoder
from vis_gaussian import render_projections
from models.dino_feat import DinoFeat
from models.VGGW import L2_norm
from models.bev_net import BEVNet

to_pil_image = transforms.ToPILImage()
EPS = utils.EPS


class Model(nn.Module):
    def __init__(self, args, device=None):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.device = device

        self.level = sorted([int(item) for item in args.level.split('_')])
        self.N_iters = args.N_iters
        self.channels = [int(item) for item in self.args.channels.split('_')]
        self.gs_channels = [64, 32, 4]

        self.SatFeatureNet = VGGUnet(self.level, self.gs_channels)
        self.gaussian_encoder = GaussianEncoder()
        self.grd_decoder = GrdDecoder()
        self.lpips = LPIPS(net="vgg")
        self.near = torch.ones(args.batch_size, 1).to(device) * 0.5
        self.far = torch.ones(args.batch_size, 1).to(device) * 160
        # self.dino_feat = DinoFeat()
        convert_to_buffer(self.lpips, persistent=False)

        if self.args.proj == 'CrossAttn':
            self.Dec4 = Decoder4(self.channels[0])
            self.Dec2 = Decoder2(self.channels[0:2])
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)
        # self.gaussian_proj = FeatureHead(64)
        if self.args.share:
            self.FeatureForT = VGGUnet(self.level, self.gs_channels)
        else:
            self.GrdFeatureForT = VGGUnet(self.level, self.gs_channels)
            self.SatFeatureForT = VGGUnet(self.level, self.gs_channels)

        self.meters_per_pixel = {}
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel[level] = meter_per_pixel * (2 ** (3 - level))

        self.TransRefine = TransOptimizerG2SP_V1(self.channels)

        self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
        self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)


        self.masks = {}
        for level in range(4):
            A = 512 / 2**(3-level)
            XYZ_1 = self.sat2world(A)  # [ sidelength,sidelength,4]

            B = 1
            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

            ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                          [0.0000, 482.7076, 125.0034],
                                          [0.0000, 0.0000, 1.0000]]],
                                        dtype=torch.float32, requires_grad=True, device=self.device)
            ori_grdH, ori_grdW = 256, 1024
            H, W = ori_grdH, ori_grdW

            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, ori_camera_k, H, W,
                                                       ori_grdH, ori_grdW)
            # [B, H, W, 2], [B, H, W, 1]
            self.masks[level] = mask[:, :, :, 0]

        # if self.args.use_uncertainty:
        #     self.uncertain_net = Uncertainty(self.channels)
        # self.bev_net = BEVNet()
        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def grd_feat_projection(self, grd_img_left, grd_feat, grd_depth, left_camera_k, deterministic=False):
        # initial
        self.camera_k = left_camera_k.clone()
        self.camera_k[:, :1, :] = self.camera_k[:, :1, :] / grd_depth.shape[2]  # original size input into feature get network/ output of feature get network
        self.camera_k[:, 1:2, :] = self.camera_k[:, 1:2, :] / grd_depth.shape[1]
        self.camera_k = self.camera_k.unsqueeze(1)
        self.extrinsics = torch.eye(4).to(grd_img_left.device).unsqueeze(0).repeat(grd_feat.shape[0], 1, 1).unsqueeze(1)
        # Encode the context images.
        self.gaussians = self.gaussian_encoder(
            self.grd_img_left,
            grd_feat, 
            self.camera_k, 
            self.extrinsics, 
            self.near,
            self.far, 
            deterministic
        )
        return self.gaussians

    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH,
                             ori_grdW):
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

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        H, W = uv.shape[1:-1]
        assert (H == W)

        # with torch.no_grad():
        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6) * \
               torch.greater_equal(uv[:, :, :, 0:1], torch.zeros_like(uv[:, :, :, 0:1])) * \
               torch.less(uv[:, :, :, 0:1], torch.ones_like(uv[:, :, :, 0:1]) * grd_W) * \
               torch.greater_equal(uv[:, :, :, 1:2], torch.zeros_like(uv[:, :, :, 1:2])) * \
               torch.less(uv[:, :, :, 1:2], torch.ones_like(uv[:, :, :, 1:2]) * grd_H)
        uv = uv * mask

        return uv, mask
        # return uv1

    def sat2world(self, satmap_sidelength):
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
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, level):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat, level)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        # shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def inplane_uv(self, ori_shift_u, ori_shift_v, ori_heading, satmap_sidelength):
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength

        B = ori_heading.shape[0]
        shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
        shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
        T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]

        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)

        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - satmap_sidelength / 2

        uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]

        uv = uv_0 + satmap_sidelength / 2
        return uv

    def NeuralOptimizer(self, grd_feat_dict, sat_feat_dict, B, left_camera_k=None, ori_grdH=None, ori_grdW=None):

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in self.level:
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]

                if self.args.stage == 0:
                    uv = self.inplane_uv(shift_u, shift_v, heading, sat_feat.shape[-1])
                    overhead_feat, _ = grid_sample(
                        grd_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                        uv, jac=None)
                else:
                    uv = self.inplane_uv(shift_u, shift_v, heading, sat_feat.shape[-1])
                    overhead_feat, _ = grid_sample(
                        grd_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                        uv, jac=None)
                # elif self.args.stage == 1:
                #     A = sat_feat.shape[-1]
                #     overhead_feat, _, _, _ = self.project_grd_to_map(
                #         grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                shift_u_new, shift_v_new, heading_new = self.Trans_update(
                    shift_u, shift_v, heading, overhead_feat, sat_feat, level)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        return shift_lats, shift_lons, thetas

    def forward(self, sat_align_cam, sat_map, grd_img_left, grd_depth, left_camera_k, gt_heading=None, gt_shift_u=None, gt_shift_v=None, train=False, loop=None, save_dir=None):
        '''
        rot_corr
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            grd_depth: [B, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')
        # sat_align_cam_ = transforms.ToPILImage()(sat_align_cam[0])
        # sat_align_cam_.save('sat_align_cam.png')
        #
        # uv = self.inplane_uv(gt_shift_u, gt_shift_v, gt_heading, sat_map.shape[-1])
        # sat_align_cam_trans, _ = grid_sample(
        #     sat_align_cam,
        #     uv, jac=None)
        # sat_align_cam_trans = transforms.ToPILImage()(sat_align_cam_trans[0])
        # sat_align_cam_trans.save('sat_align_cam_trans.png')


        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        self.grd_img_left = F.interpolate(grd_img_left, size=(64, 256), mode='bilinear', align_corners=False)
        self.sat_map = F.interpolate(sat_map, size=(256, 256), mode='bilinear', align_corners=False)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        if self.args.stage == 0:
            sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat_map)
            over_feat_dict, over_conf_dict = self.SatFeatureNet(sat_align_cam)
            # not sure whether mask should be appliced at image level or feature level

            shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B)

            for _, level in enumerate(self.level):
                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict[level]
                over_feat = over_feat_dict[level]
                over_conf = over_conf_dict[level]

                A = sat_feat.shape[-1]
                uv = self.inplane_uv(shift_u, shift_v, gt_heading, A)
                overhead_feat, _ = grid_sample(
                    over_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None)
                overhead_conf, _ = grid_sample(
                    over_conf * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None
                )

                crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(overhead_feat, [crop_H, crop_W])
                overhead_conf = TF.center_crop(overhead_conf, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = overhead_conf

            return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, self.masks, shift_lats, shift_lons, thetas, None

        elif self.args.stage == 1:
            decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
            grd_color = decoder_grd.color.clip(min=0, max=1) 
            test_img = to_pil_image(grd_color[0])
            test_img.save(f'grd_weak_maskfov1.png')
            _, _, H, W = grd_color.shape
            grd_depth_imgs = F.interpolate(grd_depth.unsqueeze(1), (H, W), mode='bilinear', align_corners=True).squeeze(1)
            grd_left_imgs = F.interpolate(grd_img_left, (H, W), mode='bilinear', align_corners=True)
            depth_l1_loss = F.l1_loss(decoder_grd.depth, grd_depth_imgs, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, grd_left_imgs, reduction='mean')
            lpips_loss = self.lpips.forward(grd_color, grd_left_imgs, normalize=True).mean()
            render_loss = depth_l1_loss + rgb_mse_loss * 20 + lpips_loss
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                grd2sat_gaussian_color1, grd2sat_gaussian_feat1 = render_projections(self.gaussians, (512,512), extra_label='prob')
                
                sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd2sat_gaussian_color1)

                if self.args.rotation_range > 0:
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            grd2sat_gaussian_color2, grd2sat_gaussian_feat2 = render_projections(self.gaussians, (128,128), extra_label='prob', heading=heading)
            test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov1.png')

            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_img_left, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')

            # test_img = to_pil_image(sat_map[0].clip(min=0, max=1))
            # test_img.save(f'origin_sat.png')

            if self.args.share:
                grd_feat_dict_forT, grd_conf_dict_forT = L2_norm(grd2sat_gaussian_feat2), torch.ones_like(grd2sat_gaussian_feat2, device=grd2sat_gaussian_feat2.device)
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd2sat_gaussian_color)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)

            grd_uv_dict = {}
            mask_dict = {}
            level = 1
            # self.level = level
            sat_feat = sat_feat_dict_forT[level]
            satmap_sidelength = sat_feat.shape[-1]
            XYZ_1 = self.sat2world(satmap_sidelength)                
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, left_camera_k, ori_grdH, ori_grdW,
                                                            ori_grdH, ori_grdW)
            g2s_feat_dict[level] = grd_feat_dict_forT * mask.permute(0, 3, 1, 2)
            g2s_conf_dict[level] = grd_conf_dict_forT * mask.permute(0, 3, 1, 2)
            grd_uv_dict[level] = uv
            mask_dict[level] = mask

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss

        elif self.args.stage == 2:

            level = self.level[0]
            grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
            self.grd_img_left = self.grd_img_left.unsqueeze(1)

            grd_gaussian = self.grd_feat_projection(self.grd_img_left, grd_feat_dict_forT[level].unsqueeze(1), grd_depth, left_camera_k)
            
            decoder_grd = self.grd_decoder.forward(
                grd_gaussian,     # Sample from variational Gaussians
                self.extrinsics,
                self.camera_k,
                self.near,
                self.far,
                (256, 1024),
            )

            grd2sat_gaussian_color, _ = render_projections(grd_gaussian, (512,512))
            test_img = to_pil_image(grd2sat_gaussian_color[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov2.png')
            test_img = to_pil_image(decoder_grd.color[0,0].clip(min=0, max=1))
            test_img.save(f'grd_weak_maskfov2.png')            
            with torch.no_grad():
                if self.args.rotation_range > 0:
                    sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat_map)
                    over_feat_dict, over_conf_dict = self.SatFeatureNet(grd2sat_gaussian_color)
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B)
                else:
                    thetas = torch.zeros([B, 1, 1], dtype=torch.float32, requires_grad=False, device=sat_map.device)
            
            return None, None, decoder_grd, None, None, None, None, thetas, None
        

        elif self.args.stage == 3:
            # sat_feat_dict = {}
            # sat_conf_dict = {}
            self.grd_img_left = F.interpolate(grd_img_left, size=(64, 256), mode='bilinear', align_corners=False)
            level = self.level[0]
            if self.args.share:
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
                grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd_img_left)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)
            
            # grd_feat_dict_forT = self.dino_feat(self.grd_img_left)
            # grd_conf_dict_forT = torch.ones_like(grd_feat_dict_forT, device=grd_feat_dict_forT.device)
            # sat_feat_dict_forT = self.dino_feat(self.sat_map)
            # sat_conf_dict_forT = torch.ones_like(sat_feat_dict_forT, device=sat_feat_dict_forT.device)
            
            decoder_grd = self.grd_feat_projection(self.grd_img_left, grd_feat_dict_forT[level], grd_depth, left_camera_k)
            grd_color = decoder_grd.color
            test_img = to_pil_image(grd_color[0])
            test_img.save(f'grd_weak_maskfov3_noshare.png')
            # _, _, H, W = grd_color.shape
            # grd_depth_imgs = F.interpolate(grd_depth.unsqueeze(1), (H, W), mode='bilinear', align_corners=True).squeeze(1)
            # grd_left_imgs = F.interpolate(grd_img_left, (H, W), mode='bilinear', align_corners=True)
            # depth_l1_loss = F.l1_loss(decoder_grd.depth, grd_depth_imgs, reduction='mean')
            # rgb_mse_loss = F.mse_loss(decoder_grd.color, grd_left_imgs, reduction='mean')
            # lpips_loss = self.lpips.forward(grd_color, grd_left_imgs, normalize=True).mean()
            render_loss = torch.tensor(0.0).to(grd_color.device)
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                grd2sat_gaussian_color1, grd2sat_gaussian_feat1 = render_projections(self.gaussians, (512,512), extra_label='prob')
                
                sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd2sat_gaussian_color1)

                if self.args.rotation_range > 0:
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            grd2sat_gaussian_color2, grd2sat_gaussian_feat2 = render_projections(self.gaussians, (128,128), extra_label='prob', heading=heading)
            test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov3_noshare.png')
            # single_features_to_RGB(grd2sat_gaussian_feat2)
            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_img_left, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')

            # test_img = to_pil_image(sat_map[0].clip(min=0, max=1))
            # test_img.save(f'origin_sat.png')

            mask_dict = {}
            sat_feat = sat_feat_dict_forT[level]
            satmap_sidelength = sat_feat.shape[-1]
            
            mask = (grd2sat_gaussian_feat2 != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
            
            g2s_feat_dict[level] = grd2sat_gaussian_feat2
            g2s_conf_dict[level] = torch.ones_like(grd2sat_gaussian_feat2, device=grd2sat_gaussian_feat2.device) * mask.permute(0, 3, 1, 2)
            mask_dict[level] = mask

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss


        elif self.args.stage == 4:
            # sat_feat_dict = {}
            # sat_conf_dict = {}
            self.grd_img_left = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
            
            level = self.level[0]
            if self.args.share:
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
                grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd2sat_gaussian_color)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)
            
            # grd_feat_dict_forT = self.dino_feat(self.grd_img_left)
            # grd_conf_dict_forT = torch.ones_like(grd_feat_dict_forT, device=grd_feat_dict_forT.device)
            # sat_feat_dict_forT = self.dino_feat(self.sat_map)
            # sat_conf_dict_forT = torch.ones_like(sat_feat_dict_forT, device=sat_feat_dict_forT.device)
            
            decoder_grd = self.grd_feat_projection(self.grd_img_left, grd_feat_dict_forT[level], grd_depth, left_camera_k)
            grd_color = decoder_grd.color
            test_img = to_pil_image(grd_color[0])
            test_img.save(f'grd_weak_maskfov4.png')
            _, _, H, W = grd_color.shape
            grd_depth_imgs = F.interpolate(grd_depth.unsqueeze(1), (H, W), mode='bilinear', align_corners=True).squeeze(1)
            grd_left_imgs = F.interpolate(grd_img_left, (H, W), mode='bilinear', align_corners=True)
            depth_l1_loss = F.l1_loss(decoder_grd.depth, grd_depth_imgs, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, grd_left_imgs, reduction='mean')
            lpips_loss = self.lpips.forward(grd_color, grd_left_imgs, normalize=True).mean()
            render_loss = depth_l1_loss + rgb_mse_loss * 20 + lpips_loss
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                if self.args.rotation_range > 0:
                    grd2sat_gaussian_color1, grd2sat_gaussian_feat1 = render_projections(self.gaussians, (512,512), extra_label='prob')
                
                    sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                    grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd2sat_gaussian_color1)
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    thetas = heading.unsqueeze(1)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            grd2sat_gaussian_color2, grd2sat_gaussian_feat2 = render_projections(self.gaussians, (256,256), extra_label='prob', heading=heading)
            test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov4.png')
            # single_features_to_RGB(grd2sat_gaussian_feat2)
            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_img_left, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')

            # test_img = to_pil_image(sat_map[0].clip(min=0, max=1))
            # test_img.save(f'origin_sat.png')

            mask_dict = {}
            sat_feat = sat_feat_dict_forT[level]
            satmap_sidelength = sat_feat.shape[-1]
            XYZ_1 = self.sat2world(satmap_sidelength)                
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, left_camera_k, ori_grdH, ori_grdW,
                                                            ori_grdH, ori_grdW)
            
            # mask = (grd2sat_gaussian_feat2 != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
            
            g2s_feat_dict[level] = grd2sat_gaussian_feat2 * mask.permute(0, 3, 1, 2)
            g2s_conf_dict[level] = torch.ones_like(grd2sat_gaussian_feat2, device=grd2sat_gaussian_feat2.device) * mask.permute(0, 3, 1, 2)
            mask_dict[level] = mask

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss

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

