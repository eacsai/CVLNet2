import torch.nn as nn
import torch
import plotly.graph_objects as go
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

from jaxtyping import Float
from torch import Tensor

from models.feature_extractor import FeatureExtractor
from models.bev_net import BEVNet
from models.dpt_single import DPT
from models.dino import DINO
from build_gaussians import build_covariance, map_pdf_to_opacity
from cuda_splatting import render_cuda
from ply_export import export_ply
from backbone.backbone_dino import BackboneDino
to_pil_image = transforms.ToPILImage()

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

class Model(nn.Module):
    def __init__(self, args, direct_map = False, use_bn = False):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.level = args.level
        self.d_sh = 25
        self.SatFeatureNet = VGGUnet(self.level)
        self.backbone = BackboneDino()
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
                83,
            ),
        )
        feature_dim = 256
        self.gaussian_param_head = nn.Conv2d(64, 3, 1, bias=False)
        self.dino_feat = DINO()
        self.dpt = DPT(self.dino_feat.feat_dim, output_dim=feature_dim)
        # self.sat_dpt = DPT(self.dino_feat.feat_dim)
        self.input_merger = nn.Sequential(
            # nn.Conv2d(256+3+3+1, 256, kernel_size=3, padding=1),
            # nn.Conv2d(3+6, 256, 7, 1, 3),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(feature_dim, 83, kernel_size=1),
        )
        self.map_pdf_to_opacity = map_pdf_to_opacity
        self.sat_out_conv = nn.Sequential(
            # nn.Conv2d(128, 128, 3, padding=1, bias=False),
            # nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
        )
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, 25 + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        
        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        
        self.image_encoder = FeatureExtractor()
        self.bev_net = BEVNet()
        torch.autograd.set_detect_anomaly(True)

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
    
    def gaussian_map(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        grd_img_left = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        grd_pts3d, gt_depth = self.forward_project(grd_img_left, left_camera_k, grd_depth, ori_grdH, ori_grdW)
        # sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        # sat_feat = sat_feat_list[-1]
        # # grd_feat = self.image_encoder(grd_img_left)["feature_maps"][0]
        # with torch.no_grad():
        #     # dino
        #     grd_feat = self.dino_feat(grd_img_left)
        #     if isinstance(grd_feat, (tuple, list)):
        #         grd_feats = [_f.detach() for _f in grd_feat]
        
        # plan A
        # grd_feat = self.dpt(grd_feats)
        # direct_grd_feat = self.input_merger(grd_img_left)
        # gaussians = self.head(grd_feat + direct_grd_feat).permute(0, 2, 3, 1)
        
        # plan B
        grd_feat = self.backbone(grd_img_left).permute(0, 2, 3, 1)
        grd_feat = self.backbone_projection(grd_feat).permute(0, 3, 1, 2)

        skip = self.high_resolution_skip(grd_img_left)
        grd_feat = grd_feat + skip
        grd_feat = rearrange(grd_feat, "b c h w -> b (h w) c")
        gaussians = self.to_gaussians(grd_feat)
        gaussians = rearrange(gaussians, "b (h w) c -> b h w c", h=ori_grdH, w=ori_grdW)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        opacities = self.map_pdf_to_opacity(densities)
        raw_gaussians = gaussians[..., 1:]
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        harmonics = sh * self.sh_mask.to(sh.device)

        covariances = build_covariance(scales, rotations)

        final_gaussians = Gaussians(
            rearrange(
                grd_pts3d,
                "b h w xyz -> b (h w) xyz",
            ),
            rearrange(
                covariances,
                "b h w i j -> b (h w) i j",
            ),
            rearrange(
                harmonics,
                "b h w c d_sh -> b (h w) c d_sh",
            ),
            rearrange(
                opacities,
                "b h w spp -> b (h w spp)",
            ),
        )
        near = torch.ones(B).to(grd_pts3d.device) * 0.1
        far = torch.ones(B).to(grd_pts3d.device) * 80
        extrinsics = torch.eye(4).to(grd_pts3d.device).unsqueeze(0).repeat(B, 1, 1)
        background_color = torch.zeros(B, 3).float().to(grd_pts3d.device)
        color, depth = render_cuda(
            extrinsics,
            left_camera_k,
            near,
            far,
            (ori_grdH, ori_grdW),
            background_color,
            final_gaussians.means,
            final_gaussians.covariances,
            final_gaussians.harmonics,
            final_gaussians.opacities,
            scale_invariant=True,
        )
        rgb_mse_loss = F.mse_loss(color, grd_img_left, reduction='mean')
        # depth_l1_loss = F.l1_loss(depth, gt_depth.squeeze(-1), reduction='mean')
        loss = rgb_mse_loss
        # test_img = to_pil_image(color[0])
        # test_img.save('test.png')
        # test_img = to_pil_image(grd_img_left[0])
        # test_img.save('gt.png')

        # ply_path = Path(f"test.ply")
        # visualization_dump={}
        # visualization_dump["scales"] = rearrange(
        #     scales, "b h w xyz -> b (h w) xyz"
        # )
        # visualization_dump["rotations"] = rearrange(
        #     rotations, "b h w xyzw -> b (h w) xyzw"
        # )
        # export_ply(
        #     torch.eye(4),
        #     final_gaussians.means[0],
        #     visualization_dump["scales"][0],
        #     visualization_dump["rotations"][0],
        #     final_gaussians.harmonics[0],
        #     final_gaussians.opacities[0],
        #     ply_path,
        # )
        return loss
