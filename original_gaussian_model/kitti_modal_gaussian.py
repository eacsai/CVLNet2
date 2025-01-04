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
import torch.optim as optim
from itertools import chain

from jaxtyping import Float
from torch import Tensor
from lpips import LPIPS

from ply_export import export_ply
from gaussian.build_gaussians import *
from vis_gaussian import render_projections
from loss.lpips import convert_to_buffer
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.autoencoder_kl import AutoencoderKL
from gaussian.pix2pix import DiscriminatorPatchGan
from gaussian.encoder import GaussianEncoder, VariationalGaussians
from gaussian.decoder import GrdDecoder
from gaussian.local_loss import MutilLocalLoss
import data_utils
from VGG import VGGUnet, L2_norm, Encoder, Decoder
from gaussian.gaussian_feature_extractor import GSDownSample, GSUpSample
from visualize import sat_features_to_RGB, single_features_to_RGB

to_pil_image = transforms.ToPILImage()
# original_raw_Lpips_step = 50000
raw_Lpips_step = 20000

def print_grad(grad):
    print("Gradient shape:", grad.shape)

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch 3 height width"]
    feature_posterior: Union[DiagonalGaussianDistribution, None]
    mask: Float[Tensor, "batch height width"]
    depth: Float[Tensor, "batch height width"]

def get_integer(f: Fraction) -> int:
    assert f.denominator == 1, "Fraction is not integer"
    return f.numerator

class Model(nn.Module):
    def __init__(self, args, device, method='down'):  # device='cuda:0',
        super(Model, self).__init__()
        self.device = device
        self.args = args
        self.d_color_sh = 25
        self.d_feature_sh = 9
        # self.global_step = 0
        self.global_step = raw_Lpips_step * 2
        self.n_feature_channels = 8
        self.variational = "gaussians"
        self.method = method
        self.supersampling_factor = 8
        self.level = args.level

        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
            
        self.gaussian_encoder = GaussianEncoder()
        self.grd_decoder = GrdDecoder()
        # self.sat_encoder = AutoencoderKL()
        self.sat_encoder = VGGUnet(self.level)
        if method == "down":
            self.grd_encoder = VGGUnet(self.level)
            # self.grd_encoder = GSDownSample(self.level)
        else:
            self.grd_encoder = GSUpSample()
        # self.grd_encoder = FeatureExtractor()
        self.discriminator = DiscriminatorPatchGan()
        self.autoencoder = AutoencoderKL()
        self.lpips = LPIPS(net="vgg")
        self.local_loss = MutilLocalLoss(args.shift_range_lat, args.shift_range_lon, args.rotation_range)
        convert_to_buffer(self.lpips, persistent=False)

    def grd_projection(self, grd_img_left, grd_depth, left_camera_k, deterministic=False) -> DecoderOutput:
        # initial
        self.camera_k = left_camera_k.clone()
        self.camera_k[:, :1, :] = self.camera_k[:, :1, :] / grd_depth.shape[2]  # original size input into feature get network/ output of feature get network
        self.camera_k[:, 1:2, :] = self.camera_k[:, 1:2, :] / grd_depth.shape[1]
        
        grd_depth = grd_depth.unsqueeze(-1)
        self.grd_depth = F.interpolate(grd_depth.permute(0,3,1,2), size=(128, 512), mode='bilinear', align_corners=False).permute(0,2,3,1)
        self.grd_img_left = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
        B, _, self.ori_grdH, self.ori_grdW = grd_img_left.shape
        self.extrinsics = torch.eye(4).to(grd_img_left.device).unsqueeze(0).repeat(B, 1, 1)
        near = torch.ones(B).to(grd_img_left.device) * 0.5
        far = torch.ones(B).to(grd_img_left.device) * 160
        # Encode the context images.
        self.gaussians: VariationalGaussians = self.gaussian_encoder(
            self.grd_img_left, 
            self.camera_k, 
            self.extrinsics, 
            self.global_step,
            near,
            far, 
            deterministic,
            self.variational in ("gaussians", "none"),
        )
        decoder_out: DecoderOutput = self.grd_decoder.forward(
            self.gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten(),     # Sample from variational Gaussians
            self.extrinsics,
            self.camera_k,
            near,
            far,
            (128, 512),
            return_colors=True,
            return_features=False,
        )
        return decoder_out

    def sat_prjection(self, mode="down"):
        color, latent = render_projections(self.gaussians.sample(), (512,512), extra_label='prob')
        projection_img = to_pil_image(color[0].clip(min=0, max=1))
        projection_img.save(f"sat_{mode}.png")
        nonzero_mask = color.abs().any(dim=1).float()
        masks = {}
        for level in range(4):
            scale = Fraction(1, 2 ** (3 - level))
            mask = self.rescale(nonzero_mask, scale)
            masks[level] = mask
        return color, masks


    def forward(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='local'):
        # train the grd generator
        decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
        test_img = to_pil_image(decoder_grd.color[0].clip(min=0, max=1))
        test_img.save(f'prob_{self.method}.png')
        # test_img = to_pil_image(sat_map[0])
        # test_img.save(f'origin_{self.method}.png')

        if mode == 'train':
            depth_l1_loss = F.l1_loss(decoder_grd.depth.unsqueeze(-1), self.grd_depth, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, self.grd_img_left, reduction='mean')
            if self.global_step >= raw_Lpips_step:
                lpips_loss = self.lpips.forward(decoder_grd.color, self.grd_img_left, normalize=True).mean()
            else:
                lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
            if self.global_step >= raw_Lpips_step*2:
                grd_color, grd_mask = self.sat_prjection(self.method)
                grd_feat_list, grd_conf_list = self.grd_encoder(grd_color)
                sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
                local_loss = self.local_loss(grd_feat_list, sat_feat_list, grd_mask, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
                self.local_loss_value = local_loss
            else:
                local_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
                self.local_loss_value = rgb_mse_loss * 20 + depth_l1_loss + lpips_loss
            total_loss = rgb_mse_loss * 20 + depth_l1_loss + local_loss * 20 + lpips_loss
            return total_loss
        elif mode == 'test':
            grd_color, grd_mask = self.sat_prjection(self.method)
            grd_feat_list, grd_conf_list = self.grd_encoder(grd_color)
            sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
            pred_u1, pred_v1 = self.local_loss(grd_feat_list, sat_feat_list, grd_mask, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
            return pred_u1, pred_v1

    def rescale(
        self,
        x: Float[Tensor, "... height width"], 
        scale_factor: Fraction
    ) -> Float[Tensor, "... downscaled_height downscaled_width"]:
        batch_dims = x.shape[:-2]
        spatial = x.shape[-2:]
        size = self.get_scaled_size(scale_factor, spatial)
        return resize(x.view(-1, *spatial), size=size, antialias=True).view(*batch_dims, *size)
    
    def get_scaled_size(self, scale: Fraction, size: Iterable[int]) -> Tuple[int, ...]:
        return tuple(get_integer(scale * s) for s in size)