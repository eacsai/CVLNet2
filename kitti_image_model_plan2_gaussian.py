import torch.nn as nn
import torch
# import plotly.graph_objects as go
from VGG import VGGUnet, L2_norm, Encoder, Decoder
import data_utils
from jacobian import grid_sample
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformer import CrossAttention
from torchvision import transforms
import matplotlib.pyplot as plt
from visualize import *
from einops import einsum, rearrange
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
from typing import Iterable, Tuple
from torchvision.transforms.functional import resize

from jaxtyping import Float
from torch import Tensor
from lpips import LPIPS
from gaussian.build_gaussians import build_covariance, map_pdf_to_opacity
from cuda_splatting import render_cuda
from gaussian.latent_splat import render_cuda, RenderOutput
# from gaussian.nopo_cuda_splatting import render_cuda

from ply_export import export_ply
from backbone.backbone_dino import BackboneDino
from depth_predictor.depth_predictor_monocular import DepthPredictorMonocular
from gaussian.build_gaussians import *
from gaussian.gaussian_adapter import get_world_rays, rotate_sh
from vis_gaussian import render_projections
from loss.lpips import convert_to_buffer
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.autoencoder_kl import AutoencoderKL

to_pil_image = transforms.ToPILImage()
Lpips_step = 15000

def print_grad(grad):
    print("Gradient shape:", grad.shape)

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Float[Tensor, "batch gaussian 3 color_d_sh"]
    feature_harmonics: Float[Tensor, "batch gaussian 4 feature_d_sh"]

@dataclass
class VariationalGaussians(Gaussians):
    feature_harmonics: DiagonalGaussianDistribution | None = None

    def _to_gaussians(self, feature_harmonics: Float[Tensor, "batch gaussian channels d_feature_sh"]) -> Gaussians:
        return Gaussians(self.means, self.covariances, self.opacities, self.color_harmonics, feature_harmonics)
    def flatten(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.params)
    def mode(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.mode())
    def sample(self) -> Gaussians:
        return self._to_gaussians(self.feature_harmonics.sample())

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
    def __init__(self, args, direct_map = False, use_bn = False):  # device='cuda:0',
        super(Model, self).__init__()
        self.global_step = 0
        self.args = args
        self.d_color_sh = 25
        self.d_feature_sh = 9
        self.n_feature_channels = 8
        self.backbone = BackboneDino()
        self.autoencoder = AutoencoderKL()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, 128),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                156,
            ),
        )
        self.gaussian_param_head = nn.Conv2d(64, 3, 1, bias=False)
        self.supersampling_factor = 8
        self.input_merger = nn.Sequential(
            # nn.Conv2d(256+3+3+1, 256, kernel_size=3, padding=1),
            # nn.Conv2d(3+6, 256, 7, 1, 3),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        
        self.depth_predictor = DepthPredictorMonocular()

        self.map_pdf_to_opacity = map_pdf_to_opacity

        self.register_buffer(
            "color_sh_mask",
            torch.ones((self.d_color_sh,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_sh_mask",
            torch.ones((self.d_feature_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, 4 + 1):
            self.color_sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        
        for degree in range(1, 2 + 1):
            self.feature_sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward_project(self, image_tensor, camera_k, depth, ori_grdH=256, ori_grdW=1024):
        image_rgb = image_tensor.clone()
        B, C, grd_H, grd_W = image_rgb.shape
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
        xyz_grd = xyz_w

        # # 可视化生成3D点云
        # xyz_grd_point = xyz_grd[1].view(-1, 3).to('cpu').detach().numpy()
        # colors = image_rgb[1].permute(1,2,0).view(-1, 3).to('cpu').detach().numpy() * 255
        # # 创建颜色字符串列表
        # colors_rgb = ['rgb({}, {}, {})'.format(r, g, b) for r, g, b in colors]

        # fig = go.Figure(data=[go.Scatter3d(
        #     x=xyz_grd_point[:, 0],
        #     y=xyz_grd_point[:, 1],
        #     z=xyz_grd_point[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=3,
        #         color=colors_rgb,  # 设置颜色
        #     )
        # )])
        # fig.update_layout(scene=dict(
        #     xaxis=dict(title='X', tick0=0, dtick=1),
        #     yaxis=dict(title='Y', tick0=0, dtick=1),
        #     zaxis=dict(title='Z', tick0=0, dtick=1)
        # ))
        # fig.show()

        return xyz_grd, depth
    
    def gaussian_init(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        # initial
        self.camera_k = left_camera_k.clone()
        self.camera_k[:, :1, :] = self.camera_k[:, :1, :] / grd_depth.shape[2]  # original size input into feature get network/ output of feature get network
        self.camera_k[:, 1:2, :] = self.camera_k[:, 1:2, :] / grd_depth.shape[1]
        
        grd_depth = grd_depth.unsqueeze(-1)
        self.grd_depth = F.interpolate(grd_depth.permute(0,3,1,2), size=(128, 512), mode='bilinear', align_corners=False).permute(0,2,3,1)
        grd_img_left = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
        B, _, self.ori_grdH, self.ori_grdW = grd_img_left.shape
        self.extrinsics = torch.eye(4).to(grd_img_left.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Encode the context images.
        grd_feat = self.backbone(grd_img_left)
        grd_feat = rearrange(grd_feat, "b c h w -> b h w c").contiguous()
        grd_feat = self.backbone_projection(grd_feat)
        grd_feat = rearrange(grd_feat, "b h w c -> b c h w").contiguous()
        
        skip = self.high_resolution_skip(grd_img_left)
        grd_feat = grd_feat + skip

        # Sample depths from the resulting features.
        self.grd_feat = rearrange(grd_feat, "b c h w -> b (h w) c").contiguous()  
        self.near = torch.ones(B).to(grd_feat.device) * 0.5
        self.far = torch.ones(B).to(grd_feat.device) * 400
        self.grd_img_left = grd_img_left

    def gaussian_map(self, deterministic=False, variational="gaussians", return_color=True, return_features=False):
        gaussians = self.to_gaussians(self.grd_feat).unsqueeze(-2)
        self.variational = variational in ("gaussians", "none")
        depths, densities = self.depth_predictor.forward(
            self.grd_feat,
            self.near,
            self.far,
            deterministic,
            1 if deterministic else 3,
        )

        # transform densities to opacities
        gpp = 1 if deterministic else 3
        opacities = self.map_pdf_to_opacity(densities) / gpp

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((self.ori_grdH, self.ori_grdW), self.grd_feat.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").contiguous()

        # gaussians = rearrange(gaussians, "b (h w) c -> b h w c", h=ori_grdH, w=ori_grdW)
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((self.ori_grdW, self.ori_grdH), dtype=torch.float32, device=self.grd_feat.device)
        coordinates = xy_ray + (offset_xy - 0.5) * pixel_size
        
        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, self.extrinsics[:, None, None, :, :], self.camera_k[:, None, None, :, :])
        grd_pts3d = directions * depths[..., None]

        # Get the raw gaussians.
        raw_gaussians = gaussians[..., 2:]

        # Split raw_gaussians into scales, rotations, color_sh and feature_sh.
        scales, rotations, color_sh, feature_sh \
            = raw_gaussians.split((3, 4, 3 * self.d_color_sh, self.n_feature_channels * self.d_feature_sh), dim=-1)

        # Map scale features to valid scale range.
        scale_min = 0.5
        scale_max = 15.0
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        multiplier = self.get_scale_multiplier(self.camera_k, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None, None, None]
        # scales = scales * depths * multiplier[:, None, None, None]

        # grd_pts3d, gt_depth = self.forward_project_v2(coordinates, left_camera_k, depths.squeeze(-1))

        # Compute the covariance matrices.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        covariances = build_covariance(scales, rotations)
        rotations = rotations.broadcast_to((*scales.shape[:-1], 4))

        # Rotate the color and feature harmonics.
        c2w_rotations = self.extrinsics[..., :3, :3]
        color_sh = rearrange(color_sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3).contiguous()
        color_sh = color_sh.broadcast_to((*depths.shape, 3, self.d_color_sh)) * self.color_sh_mask.to(color_sh.device)
        feature_sh = rearrange(feature_sh, "... (c d_sh) -> ... c d_sh", c=self.n_feature_channels).contiguous()
        feature_sh = feature_sh.broadcast_to((*depths.shape, self.n_feature_channels, self.d_feature_sh)) * self.feature_sh_mask
        color_harmonics = rotate_sh(color_sh, c2w_rotations[..., None, None, None, :, :])
        feature_harmonics = rotate_sh(feature_sh, c2w_rotations[..., None, None, None, :, :])
        gaussian_features = rearrange(
            feature_harmonics,
            "b r spp c d_f_sh -> b (r spp) c d_f_sh",
        )
        gaussian_features = DiagonalGaussianDistribution(
            **{"params" if self.variational else "mean": gaussian_features},
            dim=-2
        )
        background_color = torch.zeros(self.grd_feat.shape[0], 3).float().to(grd_pts3d.device)
        variational_gaussians = VariationalGaussians(
            rearrange(
                grd_pts3d,
                "b r spp xyz -> b (r spp) xyz",
            ),                
            rearrange(
                covariances,
                "b r spp i j -> b (r spp) i j",
            ),
            rearrange(
                opacities,
                "b r spp -> b (r spp)",
            ),
            rearrange(
                color_harmonics,
                "b r spp c d_c_sh -> b (r spp) c d_c_sh",
            ),
            gaussian_features
        )
        if self.variational:
            final_gaussians: Gaussians = variational_gaussians.sample()
        else:
            final_gaussians: Gaussians = variational_gaussians.flatten()
        color_sh = final_gaussians.color_harmonics if return_color and final_gaussians.color_harmonics is not None else None
        feature_sh = final_gaussians.feature_harmonics if return_features and final_gaussians.feature_harmonics is not None else None
        rendered: RenderOutput = render_cuda(
            self.extrinsics,
            self.camera_k,
            self.near,
            self.far,
            (128, 512),
            background_color,
            final_gaussians.means,
            final_gaussians.covariances,
            final_gaussians.opacities,
            color_sh,
            feature_sh,
        )
        out = self.render_to_decoder_output(rendered, self.extrinsics.shape[0])
        # color.register_hook(print_grad)   
        latent = out.feature_posterior.sample()
        z = self.rescale(latent, Fraction(1, self.supersampling_factor))
        skip_z = torch.cat((out.color, latent), dim=-3)
        dec = self.autoencoder.decode(z, skip_z)
        if not deterministic:    
            rgb_mse_loss = F.mse_loss(out.color, self.grd_img_left, reduction='mean')
            depth_l1_loss = F.l1_loss(out.depth.unsqueeze(-1), self.grd_depth, reduction='mean')
            if self.global_step > Lpips_step:
                lpips_loss = self.lpips.forward(
                    out.color,
                    self.grd_img_left,
                    normalize=True,
                )
            else:
                lpips_loss = torch.tensor(0, dtype=torch.float32, device=out.color.device)
            loss = rgb_mse_loss * 20 + depth_l1_loss + lpips_loss.mean()
            test_img = to_pil_image(out.color[0].clip(min=0, max=1))
            test_img.save('probabilistic_test.png')
            test_img = to_pil_image(self.grd_img_left[0])
            test_img.save('gt.png')
            render_projections(final_gaussians, (256,256), extra_label='prob')
            # ply_path = Path(f"test.ply")
            # visualization_dump={}
            # visualization_dump["scales"] = scales.reshape(B,-1,3)
            # visualization_dump["rotations"] = rotations.reshape(B,-1,4)
            # export_ply(
            #     torch.eye(4),
            #     final_gaussians.means[0],
            #     visualization_dump["scales"][0],
            #     visualization_dump["rotations"][0],
            #     final_gaussians.harmonics[0],
            #     final_gaussians.opacities[0],
            #     ply_path,
            # )

            # # 旋转矩阵 (绕 x 轴旋转 -90°)
            # rotation = torch.tensor([
            #     [1, 0,  0],
            #     [0, 0, -1],
            #     [0, 1,  0]
            # ], dtype=torch.float32)

            # # 平移向量 (0, 0, 0)
            # translation = torch.tensor([0, 0, 0], dtype=torch.float32)

            # # 构造齐次变换矩阵
            # extrinsics = torch.eye(4, dtype=torch.float32)
            # extrinsics[:3, :3] = rotation
            # extrinsics[:3, 3] = translation
            # extrinsics = extrinsics.unsqueeze(0).repeat(B, 1, 1)
            return loss
        else:
            test_img = to_pil_image(out.color[0].clip(min=0, max=1))
            test_img.save('deterministic_test.png')
            render_projections(final_gaussians, (256,256), extra_label='deterministic_test')

    def forward(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        self.gaussian_init(sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode)
        self.global_step = self.global_step + sat_map.shape[0]
        return self.gaussian_map()

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)
    
    def render_to_decoder_output(
        self,
        render_output: RenderOutput,
        b: int,
    ) -> DecoderOutput:
        if render_output.feature is not None:
            features = render_output.feature
            # NOTE background feature = 0 = mean = logvar (of normal distribution)
            mean, logvar = features, (1-rearrange(render_output.mask.detach(), "b h w -> b () h w", b=b)).log().expand_as(features)
            feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        else:
            feature_posterior = None
        return DecoderOutput(
            color=render_output.color if render_output.color is not None else None,
            feature_posterior=feature_posterior,
            mask=render_output.mask,
            depth=render_output.depth,
        )
    
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