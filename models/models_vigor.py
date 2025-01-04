import torch.nn as nn
import torch
import os
# import plotly.graph_objects as go
# from VGG import VGGUnet, L2_norm, Encoder, Decoder
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Tuple, Union
from torchvision.transforms.functional import resize

from jaxtyping import Float
from torch import Tensor
from lpips import LPIPS
import numpy as np

from gaussian.build_gaussians import *
from vis_gaussian import render_projections
from loss.lpips import convert_to_buffer
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution

from gaussian.encoder import GaussianEncoder
from gaussian.decoder import GrdDecoder
from gaussian.local_loss import LocalLoss
import data_utils
from VGG import VGGUnet, L2_norm, Encoder, Decoder
from gaussian.gaussian_feature_extractor import GSDownSample, GSUpSample
from depth_anything_v2.dpt import DepthAnythingV2
from models.pano_utils import *
from models.six_split import split_panorama

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

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch 3 height width"]
    feature: Union[DiagonalGaussianDistribution, None]
    depth: Float[Tensor, "batch height width"]

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
        self.d_color_sh = 25
        self.d_feature_sh = 9
        self.variational = "gaussians"
        self.level = args.level
        
        self.face_num = 10
        self.gaussian_encoder = GaussianEncoder()
        self.grd_decoder = GrdDecoder()
        self.near = torch.ones(args.batch_size, self.face_num).to(device) * 0.5
        self.far = torch.ones(args.batch_size, self.face_num).to(device) * 160

        # depth_anything_v2 = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 80})
        # depth_anything_v2.load_state_dict(torch.load('/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
        # self.depth_anything_v2 = depth_anything_v2.to(device).eval()

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        self.global_step = 0

    def forward2DoF(self, sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot=None, loop=None, save_dir=None):
        b, v, _, h, w = pers_imgs.shape
        # showDepth(pers_depths[0,0,0], pers_imgs[0,0].permute(1,2,0))
        grd_gaussian = self.gaussian_encoder(
            pers_imgs,
            None, 
            camera_k, 
            extrinsics, 
            self.near,
            self.far, 
            False,
        )

        decoder_grd = self.grd_decoder(
            grd_gaussian,     # Sample from variational Gaussians
            extrinsics,
            camera_k,
            self.near,
            self.far,
            (160,160)
            # (160 - start_height, 320),
        )
        grd_color = decoder_grd.color
        test_img = to_pil_image(grd_color[0,2].clip(min=0, max=1))
        test_img.save(f'grd_vigor_stage1.png')
        test_img = to_pil_image(pers_imgs[0,2].clip(min=0, max=1))
        test_img.save(f'real_vigor_stage1.png')
        heading = torch.zeros([sat.shape[0], 1], dtype=torch.float32, requires_grad=False, device=sat.device)

        # xyz_coords = equirectangular_to_xyz(320, 160, start_height=start_height)
        # points = torch.tensor(xyz_coords, dtype=torch.float32, requires_grad=False, device=sat.device).unsqueeze(0).repeat(sat.shape[0], 1, 1, 1)
        # points = points * decoder_grd.depth.unsqueeze(-1)
        # self.gaussians.means = points.reshape(sat.shape[0], -1, 3)

        grd2sat_gaussian_color2, _ = render_projections(grd_gaussian, (256,256), heading=heading, look_axis=2)
        test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
        test_img.save(f'sat_vigor_stage1.png')
        rgb_mse_loss = F.mse_loss(decoder_grd.color, pers_imgs, reduction='mean')
        depth_l1_loss = F.l1_loss(decoder_grd.depth, depth_imgs, reduction='mean')
        if self.global_step >= raw_Lpips_step:
            raw_lpips_loss = self.lpips.forward(decoder_grd.color, pers_imgs,  normalize=True)
        else:
            raw_lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
        
        self.render_loss = rgb_mse_loss * 20 + depth_l1_loss + raw_lpips_loss.mean()
        self.global_step = self.global_step + sat.shape[0]
        return self.render_loss
    
    def forward(self, sat_rt, sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None, loop=None, save_dir=None):
        idx = torch.tensor([2, 3, 4, 7, 9, 10, 11, 14, 16, 19], dtype=torch.int64, device=sat.device)
        # idx = torch.tensor([0,1,2,3,5], dtype=torch.int64, device=sat.device)

        # with torch.no_grad():
        #     pers_imgs, extrinsics, camera_k = split_panorama(grd, gen_res=80, device=self.device)
        #     pers_imgs = pers_imgs[:, idx]
        #     b,v,c,h,w = pers_imgs.shape
        #     camera_k = camera_k.unsqueeze(0).repeat(b, v, 1, 1)
        #     extrinsics = extrinsics[idx].unsqueeze(0).repeat(b, 1, 1, 1)            
        #     depth = self.depth_anything_v2.infer_image(pers_imgs.reshape(b*v,c,h,w), 518)
        #     depth = depth.view(b,v,h,w)
        #     mask = torch.any(pers_imgs != 0, dim=2).float()
        #     depth = depth * mask.to(depth.device)
            # showDepth(depth[0, i], tensor_to_cv2_image(pers_imgs[0, i]))
        pers_imgs = pers_imgs[:, idx]
        depth_imgs = depth_imgs[:, idx]
        camera_k = camera_k[:, idx]
        extrinsics = extrinsics[:, idx]

        loss = self.forward2DoF(sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot, loop, save_dir)
        return loss




from models.models_kitti import batch_wise_cross_corr, weak_supervise_loss, corr_for_accurate_translation_supervision


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
