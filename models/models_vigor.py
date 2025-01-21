import torch.nn as nn
import torch
import os
# import plotly.graph_objects as go
from models.VGGW import VGGUnet, L2_norm, Encoder, Decoder
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Tuple, Union
from torchvision.transforms.functional import resize

from models.pano_utils import split_panorama
from jaxtyping import Float
from torch import Tensor
from lpips import LPIPS
import numpy as np

from loss.lpips import convert_to_buffer
from gaussian.encoder import GaussianEncoder
from gaussian.decoder import GrdDecoder
from vis_gaussian import render_projections
from jacobian import grid_sample
from visualize import *
import cv2
to_pil_image = transforms.ToPILImage()
# original_raw_Lpips_step = 50000
raw_Lpips_step = 25000
# original_L1_step = 100000
L1_step = 40000
# original_refine_Lpips_step = 100000
refine_Lpips_step = 40000
# original_discriminator_loss_active_step = 125000
discriminator_loss_active_step = 60000


def get_integer(f: Fraction) -> int:
    assert f.denominator == 1, "Fraction is not integer"
    return f.numerator

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class ModelVIGOR(nn.Module):
    def __init__(self, args, device):  # device='cuda:0',
        super(ModelVIGOR, self).__init__()
        self.device = device
        self.args = args
        self.grd_res = args.grd_res
        self.level = sorted([int(item) for item in args.level.split('_')])[0]
        self.N_iters = args.N_iters
        self.channels = ([int(item) for item in self.args.channels.split('_')])

        self.gaussian_encoder = GaussianEncoder(32)
        self.grd_decoder = GrdDecoder()
        # self.near = torch.ones(args.batch_size, self.face_num).to(device) * 0.5
        # self.far = torch.ones(args.batch_size, self.face_num).to(device) * 160
        self.FeatureForT = VGGUnet(self.level, self.channels)
        self.SatFeatureNet = VGGUnet(self.level, self.channels)
        self.GrdFeatureNet = VGGUnet(self.level, self.channels)
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        self.global_step = 0

        torch.autograd.set_detect_anomaly(True)

    def forwardGS(self, sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot=None, loop=None, save_dir=None):
        b, v, c, h, w = pers_imgs.shape
        grd_res = 160
        # showDepth(depth_imgs[0,0], tensor_to_cv2_image(pers_imgs[0,0]))
        self.near = torch.ones(b, v).to(pers_imgs.device) * 0.5
        self.far = torch.ones(b, v).to(pers_imgs.device) * 160
        gs_pers_imgs = F.interpolate(pers_imgs.view(b*v, c, h, w), (grd_res, grd_res), mode='bilinear', align_corners=True).view(b, v, c, grd_res, grd_res)

        grd_gaussian = self.gaussian_encoder(
            gs_pers_imgs,
            None, 
            camera_k, 
            extrinsics, 
            self.near,
            self.far, 
            False,
        )

        # grd_gaussian = self.gaussian_encoder(
        #     pers_imgs,
        #     None, 
        #     camera_k, 
        #     extrinsics, 
        #     self.near,
        #     self.far, 
        #     False,
        # )

        decoder_grd = self.grd_decoder(
            grd_gaussian,     # Sample from variational Gaussians
            extrinsics,
            camera_k,
            self.near,
            self.far,
            (grd_res,grd_res)
            # (160 - start_height, 320),
        )
        grd_color = decoder_grd.color
        test_img = to_pil_image(grd_color[0,2].clip(min=0, max=1))
        test_img.save(f'grd_vigor_20face_{grd_res}.png')
        test_img = to_pil_image(pers_imgs[0,2].clip(min=0, max=1))
        test_img.save(f'real_vigor_20face_{grd_res}.png')
        heading = torch.zeros([sat.shape[0], 1], dtype=torch.float32, requires_grad=False, device=sat.device)

        # xyz_coords = equirectangular_to_xyz(320, 160, start_height=start_height)
        # points = torch.tensor(xyz_coords, dtype=torch.float32, requires_grad=False, device=sat.device).unsqueeze(0).repeat(sat.shape[0], 1, 1, 1)
        # points = points * decoder_grd.depth.unsqueeze(-1)
        gs_depth_imgs = F.interpolate(depth_imgs.view(b*v, 1, h, w), (grd_res, grd_res), mode='bilinear', align_corners=True).view(b, v, grd_res, grd_res)

        grd2sat_gaussian_color2, _ = render_projections(grd_gaussian, (256,256), heading=heading, look_axis=2)
        test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
        test_img.save(f'sat_vigor_20face_{grd_res}.png')
        rgb_mse_loss = F.mse_loss(decoder_grd.color, gs_pers_imgs, reduction='mean')
        depth_l1_loss = F.l1_loss(decoder_grd.depth, gs_depth_imgs, reduction='mean')
        if self.global_step >= raw_Lpips_step:
            raw_lpips_loss = self.lpips.forward(decoder_grd.color.view(b*v, c, grd_res, grd_res), gs_pers_imgs.view(b*v, c, grd_res, grd_res),  normalize=True)
        else:
            raw_lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
        
        self.render_loss = rgb_mse_loss * 20 + depth_l1_loss + raw_lpips_loss.mean()
        self.global_step = self.global_step + sat.shape[0]
        return self.render_loss


    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        # shift_u = shift_u / np.power(2, 3 - level)
        # shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        # shift_u = shift_u / 512 * S
        # shift_v = shift_v / 512 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3-level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(-2))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_f, grd_c, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_f, uv)
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv

    # @profile
    def forward2DoF(self, sat, grd, gt_rot, meter_per_pixel):
        self.gaussian_encoder.eval()
        idx = torch.tensor([2, 3, 4, 7, 9, 10, 11, 14, 16, 19], dtype=torch.int64, device=sat.device)
        # idx = torch.tensor([0], dtype=torch.int64, device=sat.device)
        # gs_grd = F.interpolate(grd, (160, 320), mode='bilinear', align_corners=True)
        pers_imgs, extrinsics, camera_k = split_panorama(grd, gen_res=160, device=self.device)
        b,v,c,h,w = pers_imgs.shape
        camera_k = camera_k.unsqueeze(0).repeat(b, v, 1, 1)
        extrinsics = extrinsics.unsqueeze(0).repeat(b, 1, 1, 1)
        pers_imgs = pers_imgs[:, idx]
        camera_k = camera_k[:, idx]
        extrinsics = extrinsics[:, idx]

        R_transform = torch.tensor([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ], dtype=torch.float32, device=sat.device)

        # 扩展 R_transform 为 4x4 齐次变换矩阵
        R_transform_homo = torch.eye(4, dtype=torch.float32, device=sat.device)
        R_transform_homo[:3, :3] = R_transform

        # 将 extrinsics 转换到 OpenCV 坐标系
        # extrinsics @ R_transform_homo.T 实现变换
        extrinsics =  R_transform_homo @ extrinsics

        v = len(idx)
        self.near = torch.ones(b, v).to(self.device) * 0.5
        self.far = torch.ones(b, v).to(self.device) * 160
        self.sat = F.interpolate(sat, (128, 128), mode='bilinear', align_corners=True)
        
        gs_pers_imgs = F.interpolate(pers_imgs.view(b*v, c, h, w), (self.grd_res, self.grd_res), mode='bilinear', align_corners=True).view(b, v, c, self.grd_res, self.grd_res)
        
        # sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat)
        # grd_feat_dict, grd_conf_dict = self.GrdFeatureNet(grd)

        sat_feat_dict, sat_conf_dict = self.FeatureForT(sat)
        grd_feat_dict, grd_conf_dict = self.FeatureForT(pers_imgs.view(b*v, c, h, w))
        if self.grd_res == 80:
            grd_feat = F.interpolate(grd_feat_dict[self.level], scale_factor=2)
            pers_feat = grd_feat.view(b, v, -1, self.grd_res, self.grd_res)
            grd_conf = F.interpolate(grd_conf_dict[self.level], scale_factor=2)
            pers_feat = grd_conf.view(b, v, -1, self.grd_res, self.grd_res)
        else:
            pers_feat = grd_feat_dict[self.level].view(b, v, -1, self.grd_res, self.grd_res)
            pers_conf = grd_conf_dict[self.level].view(b, v, -1, self.grd_res, self.grd_res)
        # pers_feat, _, _ = split_panorama(grd_feat, gen_res=self.grd_res, device=self.device)
        # pers_feat = pers_feat[:, idx]

        g2s_feat_dict = {}
        g2s_conf_dict = {}
        shift_u = torch.zeros([sat.shape[0]], dtype=torch.float32, requires_grad=True, device=sat.device)
        shift_v = torch.zeros([sat.shape[0]], dtype=torch.float32, requires_grad=True, device=sat.device)

        grd_gaussian = self.gaussian_encoder(
            gs_pers_imgs, 
            pers_feat, 
            pers_conf, 
            camera_k, 
            extrinsics,
            self.near,
            self.far,
            False,
        )
        
        # decoder_grd = self.grd_decoder(
        #     grd_gaussian,     # Sample from variational Gaussians
        #     extrinsics,
        #     camera_k,
        #     self.near,
        #     self.far,
        #     (self.grd_res,self.grd_res)
        #     # (160 - start_height, 320),
        # )
        
        # idx = 0
        # grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
        #     grd, None, gt_rot, shift_u, shift_v, self.level, meter_per_pixel)
        # grd_color = decoder_grd.color
        # test_img = to_pil_image(grd_color[idx,2].clip(min=0, max=1))
        # test_img.save(f'grd_vigor_s3_20face_{self.grd_res}.png')
        # test_img = to_pil_image(grd[idx].clip(min=0, max=1))
        # test_img.save(f'ori_grd_vigor_s3_20face_{self.grd_res}.png')
        # test_img = to_pil_image(pers_imgs[idx,2].clip(min=0, max=1))
        # test_img.save(f'real_vigor_s3_20face_{self.grd_res}.png')
        # test_img = to_pil_image(self.sat[idx].clip(min=0, max=1))
        # test_img.save(f'sat_vigor_s3_20face_{self.grd_res}.png')
        # test_img = to_pil_image(grd_feat_proj[idx].clip(min=0, max=1))
        # test_img.save(f'ori_vigor_s3_20face_{self.grd_res}.png')

        if self.args.rotation_range == 0:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90
            rot_range = 1
        else:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90 / self.args.rotation_range
            heading = heading + gt_rot.unsqueeze(-1)
            rot_range = self.args.rotation_range
        grd2sat_gaussian_color2, grd2sat_gaussian_feat2, grd2sat_gaussian_conf2 = render_projections(grd_gaussian, (128,128), heading=heading, rot_range=rot_range)
        grd2sat_feat2 = grd2sat_gaussian_feat2
        grd2sat_conf2 = grd2sat_gaussian_conf2

        # # vis
        # test_img = to_pil_image(grd2sat_gaussian_color2[idx].clip(min=0, max=1))
        # test_img.save(f'g2s_vigor_s3_20face_{self.grd_res}.png')
        # # vis
        # single_features_to_RGB(grd2sat_feat2, idx)
        # sat_features_to_RGB(sat_feat_dict[self.level], grd2sat_feat2, idx)
        # sat_features_to_RGB_2D_PCA(sat_feat_dict[self.level], grd2sat_feat2, idx)
        # grd_features_to_RGB_2D_PCA_concat(pers_feat)
        
        sat_feat = sat_feat_dict[self.level]
        A = sat_feat.shape[-1]
        crop_H = int(A * 0.4)
        crop_W = int(A * 0.4)
        g2s_feat = TF.center_crop(grd2sat_feat2, [crop_H, crop_W])
        # mask = (g2s_feat != 0).any(dim=1, keepdim=True).float()

        g2s_conf = TF.center_crop(grd2sat_conf2, [crop_H, crop_W])

        g2s_feat_dict[self.level] = g2s_feat
        g2s_conf_dict[self.level] = g2s_conf
        
        sat_uncer_dict = {}
        for level in range(3):
            sat_uncer_dict[level] = None
        return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict
    
    def forward(self, sat, grd, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None, loop=None, save_dir=None):
        if self.args.Supervision == 'Gaussian':
            loss = self.forwardGS(sat, grd, meter_per_pixel, gt_rot, loop, save_dir)
            return loss
        else:
            return self.forward2DoF(sat, grd, gt_rot, meter_per_pixel)
        

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
            
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
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


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, args, meters_per_pixel, GPS_error=5):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    gt_shift_u: [B]
    gt_shift_v: [B]
    meters_per_pixel: [B], corresponding to original image size
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []

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
        w = (torch.round(W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        h = (torch.round(H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        radius = (torch.ceil(GPS_error / (meters_per_pixel * np.power(2, 3 - level)))).long()
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius[b_idx])
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius[b_idx])
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius[b_idx])
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius[b_idx])
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))



def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, args):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        w = torch.round(corr_W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)
        h = torch.round(corr_H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)

        # import pdb; pdb.set_trace()
        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))



def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])

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

        denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    if args.use_uncertainty:
        denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)
    pred_u = (max_index % corr_W - corr_W / 2)
    pred_v = (max_index // corr_W - corr_H / 2)

    # if level == 3:
    #     return pred_u, pred_v, corr
    #
    # elif level == 2:
    #     return pred_u * 2, pred_v * 2, corr

    return pred_u * np.power(2, 3 - level), pred_v * np.power(2, 3 - level), corr

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