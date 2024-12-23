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
from models.gaussian_feature_extractor import GSDownSample, GSUpSample
from visualize import sat_features_to_RGB, single_features_to_RGB
import numpy as np

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
            self.masks[level] = mask[:, :, :, 0].float()

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

    def forward(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='local'):
        # train the grd generator
        decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
        test_img = to_pil_image(decoder_grd.color[0].clip(min=0, max=1))
        test_img.save(f'prob_{self.method}_maskfov.png')
        test_img = to_pil_image(sat_map[0])
        test_img.save(f'origin_{self.method}_maskfov.png')

        if mode == 'train':
            depth_l1_loss = F.l1_loss(decoder_grd.depth.unsqueeze(-1), self.grd_depth, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, self.grd_img_left, reduction='mean')
            if self.global_step >= raw_Lpips_step:
                lpips_loss = self.lpips.forward(decoder_grd.color, self.grd_img_left, normalize=True).mean()
            else:
                lpips_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
            if self.global_step >= raw_Lpips_step*2:
                grd_color, grd_latent = render_projections(self.gaussians.sample(), (512,512), extra_label='prob')
                grd_feat_list, grd_conf_list = self.grd_encoder(grd_color)
                sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
                local_loss = self.local_loss(grd_feat_list, sat_feat_list, self.masks, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
                self.local_loss_value = local_loss
            else:
                local_loss = torch.tensor(0, dtype=torch.float32, device=decoder_grd.color.device)
                self.local_loss_value = rgb_mse_loss * 20 + depth_l1_loss + lpips_loss
            total_loss = rgb_mse_loss * 20 + depth_l1_loss + local_loss * 20 + lpips_loss
            return total_loss
        elif mode == 'test':
            grd_color, grd_latent = render_projections(self.gaussians.sample(), (512,512), extra_label='prob')
            grd_feat_list, grd_conf_list = self.grd_encoder(grd_color)
            sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
            pred_u1, pred_v1 = self.local_loss(grd_feat_list, sat_feat_list, self.masks, gt_shift_u, gt_shift_v, gt_heading, mode=mode)
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

            camera_height = 1.65
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
        meter_per_pixel = data_utils.get_meter_per_pixel()
        meter_per_pixel *= data_utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap