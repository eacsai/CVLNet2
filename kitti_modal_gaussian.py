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
from typing import Iterable, Tuple
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
from gaussian.local_loss import LocalLoss
import data_utils
from VGG import VGGUnet, L2_norm, Encoder, Decoder
from models.gaussian_feature_extractor import GSDownSample, GSUpSample

to_pil_image = transforms.ToPILImage()
# original_raw_Lpips_step = 50000
raw_Lpips_step = 25000
# original_L1_step = 100000
L1_step = 50000
# original_refine_Lpips_step = 100000
refine_Lpips_step = 50000
# original_discriminator_loss_active_step = 125000
discriminator_loss_active_step = 65000

sat_prjection_step = 100000

def print_grad(grad):
    print("Gradient shape:", grad.shape)

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch 3 height width"] | None
    feature_posterior: DiagonalGaussianDistribution | None
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
        self.global_step = 0
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
            self.grd_encoder = GSDownSample(self.level)
        else:
            self.grd_encoder = GSUpSample()
        # self.grd_encoder = FeatureExtractor()
        self.discriminator = DiscriminatorPatchGan()
        self.autoencoder = AutoencoderKL()
        self.lpips = LPIPS(net="vgg")
        self.local_loss = LocalLoss(args.shift_range_lat, args.shift_range_lon, args.rotation_range)
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
        sat_img = color

        projection_img = to_pil_image(sat_img[0].clip(min=0, max=1))
        projection_img.save(f"sat_proj_{mode}.png")
        if mode == 'down':
            return latent
        else:
            z = self.rescale(latent, Fraction(1, self.supersampling_factor))
            feat = self.autoencoder.feature_extractor(z)
            return feat

    def forward(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='local'):
        # train the grd generator
        decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
        test_img = to_pil_image(decoder_grd.color[0].clip(min=0, max=1))
        test_img.save(f'prob_test_{self.method}.png')

        grd_map = self.sat_prjection(self.method)
        grd_feat_list, grd_conf_list = self.grd_encoder(grd_map)

        grd_feat = grd_feat_list[-1]
        sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
        sat_feat = sat_feat_list[-1]
        if mode == 'train':
            local_loss = self.local_loss(grd_feat, sat_feat, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
            depth_l1_loss = F.l1_loss(decoder_grd.depth.unsqueeze(-1), self.grd_depth, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, self.grd_img_left, reduction='mean')
            if self.global_step >= raw_Lpips_step:
                lpips_loss = self.lpips.forward(decoder_grd.color, self.grd_img_left, normalize=True).mean()
            else:
                lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)

            total_loss = rgb_mse_loss * 20 + depth_l1_loss + local_loss * 20 + lpips_loss
            self.local_loss_value = local_loss
            return total_loss
        elif mode == 'test':
            pred_u1, pred_v1 = self.local_loss(grd_feat, sat_feat, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
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
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad