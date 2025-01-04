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
from depth_anything_v2.dpt import DepthAnythingV2

from pano_utils import *
from jaxtyping import Float
from torch import Tensor
from lpips import LPIPS
import numpy as np

from loss.lpips import convert_to_buffer
from gaussian.encoder import GaussianEncoder
from gaussian.decoder import GrdDecoder
from vis_gaussian import render_projections

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

        depth_anything_v2 = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 80})
        depth_anything_v2.load_state_dict(torch.load('/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
        self.depth_anything_v2 = depth_anything_v2.to(device).eval()


        self.face_num = 6
        self.gaussian_encoder = GaussianEncoder()
        self.grd_decoder = GrdDecoder()
        self.near = torch.ones(args.batch_size, self.face_num).to(device) * 0.5
        self.far = torch.ones(args.batch_size, self.face_num).to(device) * 160

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        self.global_step = 0

    def create_data(self, grd, save_path):

        with torch.no_grad():
            pers_imgs, extrinsics, camera_k = split_panorama(grd, gen_res=160, device=self.device)
            b,v,c,h,w = pers_imgs.shape
            camera_k = camera_k.unsqueeze(0).repeat(b, v, 1, 1)
            extrinsics = extrinsics.unsqueeze(0).repeat(b, 1, 1, 1)
            
            depth = self.depth_anything_v2.infer_image(pers_imgs.view(b*v,c,h,w), 518)
            depth = depth.view(b,v,h,w)
            mask = torch.any(pers_imgs != 0, dim=2).float()
            depth = depth * mask.to(depth.device)

            for b in range(len(save_path)):
                data_to_save = {
                    'depth_imgs': depth[b],
                    'pers_imgs': pers_imgs[b],
                    'camera_k': camera_k[b],
                    'extrinsics': extrinsics[b]
                }

                torch.save(data_to_save, save_path[b])
            # showDepth(depth[0, i], tensor_to_cv2_image(pers_imgs[0, i]))
            
        # loss = self.forward2DoF(sat, grd, pers_imgs, depth, camera_k, extrinsics, meter_per_pixel, gt_rot, loop, save_dir)
        # return loss


    def forward2DoF(self, sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot=None, loop=None, save_dir=None):
        b, v, c, h, w = pers_imgs.shape
        # showDepth(depth_imgs[0,0], tensor_to_cv2_image(pers_imgs[0,0]))
        
        # gs_pers_imgs = F.interpolate(pers_imgs.view(b*v, c, h, w), (80, 80), mode='bilinear', align_corners=True).view(b, v, c, 80, 80)

        grd_gaussian = self.gaussian_encoder(
            pers_imgs[:, :4],
            None, 
            camera_k[:, :4], 
            extrinsics[:, :4], 
            self.near[:, :4],
            self.far[:, :4], 
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
            (160,160)
            # (160 - start_height, 320),
        )
        grd_color = decoder_grd.color
        test_img = to_pil_image(grd_color[0,2].clip(min=0, max=1))
        test_img.save(f'grd_vigor_6face_160.png')
        test_img = to_pil_image(pers_imgs[0,2].clip(min=0, max=1))
        test_img.save(f'real_vigor_6face_160.png')
        heading = torch.zeros([sat.shape[0], 1], dtype=torch.float32, requires_grad=False, device=sat.device)

        # xyz_coords = equirectangular_to_xyz(320, 160, start_height=start_height)
        # points = torch.tensor(xyz_coords, dtype=torch.float32, requires_grad=False, device=sat.device).unsqueeze(0).repeat(sat.shape[0], 1, 1, 1)
        # points = points * decoder_grd.depth.unsqueeze(-1)
        # self.gaussians.means = points.reshape(sat.shape[0], -1, 3)

        grd2sat_gaussian_color2, _ = render_projections(grd_gaussian, (256,256), heading=heading, look_axis=2)
        test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
        test_img.save(f'sat_vigor_6face_160.png')
        rgb_mse_loss = F.mse_loss(decoder_grd.color, pers_imgs, reduction='mean')
        depth_l1_loss = F.l1_loss(decoder_grd.depth[:,:4], depth_imgs[:,:4], reduction='mean')
        if self.global_step >= raw_Lpips_step:
            raw_lpips_loss = self.lpips.forward(decoder_grd.color.view(b*v, c, h, w), pers_imgs.view(b*v, c, h, w),  normalize=True)
        else:
            raw_lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
        
        self.render_loss = rgb_mse_loss * 20 + depth_l1_loss + raw_lpips_loss.mean()
        self.global_step = self.global_step + sat.shape[0]
        return self.render_loss

    def forward(self, sat_rt, sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None, loop=None, save_dir=None):
        # idx = torch.tensor([2, 3, 4, 7, 9, 10, 11, 14, 16, 19], dtype=torch.int64, device=sat.device)
        idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=sat.device)

        pers_imgs = pers_imgs[:, idx]
        depth_imgs = depth_imgs[:, idx]
        camera_k = camera_k[:, idx]
        extrinsics = extrinsics[:, idx]

        loss = self.forward2DoF(sat, grd, pers_imgs, depth_imgs, camera_k, extrinsics, meter_per_pixel, gt_rot, loop, save_dir)
        return loss
