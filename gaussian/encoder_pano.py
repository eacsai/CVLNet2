from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

import open3d as o3d
import plotly.graph_objs as go

from backbone.backbone_dino import BackboneDino
from depth_predictor.depth_predictor_monocular import DepthPredictorMonocular
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.build_gaussians import sample_image_grid
from gaussian.gaussian_adapter_pano import GaussianAdapter

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Union[Float[Tensor, "batch gaussian 3 color_d_sh"], None]
    features: Float[Tensor, "batch gaussian dim"]
    confidence: Float[Tensor, "batch gaussian 1"]


def equirectangular_to_xyz(width, height, device):
    """Convert equirectangular coordinates to spherical 3D coordinates in OpenCV convention and rotate 90° around Y-axis using PyTorch."""
    
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, width, device=device)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, height, device=device)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = torch.sin(phi) * torch.cos(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.sin(theta)  # OpenCV Z: 前

    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    xyz = torch.stack((x, y, z), dim=-1).permute(1, 0, 2)  # (H, W, 3)

    # 旋转矩阵 (顺时针旋转 90 度)
    R_y_90 = torch.tensor([
        [0, 0, 1],  
        [0, 1, 0],  
        [-1, 0, 0]
    ], dtype=torch.float32, device=device)

    # 将点云展平进行矩阵乘法 (H*W, 3) x (3,3)
    xyz_rotated = xyz.reshape(-1, 3) @ R_y_90.T  # 应用旋转

    # 还原形状为 (H, W, 3)
    xyz_rotated = xyz_rotated.view(height, width, 3)

    return xyz_rotated


# def equirectangular_to_xyz(width, height, device):
#     """Convert equirectangular coordinates to spherical 3D coordinates."""
#     theta = torch.linspace(0, 2 * torch.pi, width, device=device)
#     phi = torch.linspace(0, torch.pi, height, device=device)  

#     theta, phi = torch.meshgrid(theta, phi, indexing='ij')

#     x = torch.sin(phi) * (-torch.cos(theta))
#     y = torch.sin(phi) * torch.sin(theta)
#     z = torch.cos(phi)
    
#     return torch.stack((x, y, z), dim=-1).permute(1, 0, 2)


class GaussianEncoder(nn.Module):
    def __init__(self, n_feature_channels) -> None:
        super(GaussianEncoder, self).__init__()
        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, 128),
        )
        self.depth_predictor = DepthPredictorMonocular()
        self.gaussian_adapter = GaussianAdapter(n_feature_channels)
        
        self.to_opacity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                82,
            ),
        )
        # self.to_gaussians_feat = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(
        #         128,
        #         148,
        #     ),
        # )
        # High resolution skip only required in case of now downscaling
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
    ) -> Float[Tensor, " *batch"]:
        exponent = 1.0
        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        img: Float[Tensor, "batch view channels height width"],
        grd_feat: Union[Float[Tensor, "batch view channels height width"] , None],
        grd_conf: Union[Float[Tensor, "batch view channels height width"] , None],
        camera_k: Float[Tensor, "batch view 3 3"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        real_depth: Float[Tensor, "batch 1 height width"],
        deterministic: bool = False,
    ) -> Gaussians:
        b, v, _, h, w = img.shape
        features = self.backbone(img)
        device = features.device
        h, w = features.shape[-2:]
        features = rearrange(features, "b v c h w -> b v h w c").contiguous()
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w").contiguous()

        if self.high_resolution_skip is not None:
            # Add the high-resolution skip connection.
            skip = rearrange(img, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip(skip)
            features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")
        depths, densities = self.depth_predictor.forward(
            features,
            near,
            far,
            deterministic,
            1 if deterministic else 3,
        )

        # Convert the features and depths into Gaussians.
        # xy_ray, _ = sample_image_grid((h, w), device)
        # xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = self.to_gaussians(features).unsqueeze(-2)
        # offset_xy = gaussians[..., :2].sigmoid()
        # pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        # xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        xyz_coords = equirectangular_to_xyz(w, h, device)
        coords = xyz_coords.reshape(-1, 3).unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
        means = coords.unsqueeze(-2) * depths.unsqueeze(-1)
        # 假设你有以下张量
        # image_tensor = img[:,0]  # 图像张量，形状为 [1, 3, 80, 160]
        # fake_depth = depths[:, :, :, 0].reshape(1, 1, 80, 160)  # 假设深度张量，形状为 [1, 80, 160]
        # coords_tensor = xyz_coords.unsqueeze(0) * real_depth.squeeze(1).unsqueeze(-1) # 三维坐标点张量，形状为 [1, 80, 160, 3]
        # # coords_tensor = xyz_coords.unsqueeze(0) * fake_depth.squeeze(1).unsqueeze(-1) # 三维坐标点张量，形状为 [1, 80, 160, 3]
        # # 提取 3D 坐标点 (x, y, z)
        # points = coords_tensor[0].reshape(-1, 3).cpu().detach().numpy()  # 将坐标点张量展平为 (80*160, 3)

        # # 提取颜色 (这里假设使用图像的 RGB 值作为颜色)
        # colors = image_tensor[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()  # 将图像张量展平为 (80*160, 3)

        # colors_rgb = ['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]  # 转换为字符串形式的 RGB

        # fig = go.Figure(data=[go.Scatter3d(
        #     x=points[:, 0],
        #     y=points[:, 1],
        #     z=points[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=3,
        #         color=colors_rgb,  # 设置颜色
        #     )
        # )])
        # fig.update_layout(scene=dict(
        #     xaxis=dict(title='X', tick0=0, dtick=1),  # X 轴方向正确
        #     yaxis=dict(title='Y (Down)', tick0=0, dtick=-1),  # 翻转Y轴
        #     zaxis=dict(title='Z', tick0=0, dtick=1),  # Z 轴方向正确
        #     aspectmode='cube'  # 确保XYZ比例一致
        # ))
        # # fig.show()

        # # 保存为 HTML 文件，下载后用浏览器打开
        # fig.write_html("point_cloud1.html")

        gpp = 3
        gaussians = self.gaussian_adapter.forward(
            extrinsics,
            camera_k,
            means,
            depths,
            self.map_pdf_to_opacity(densities) / gpp,
            gaussians,
            grd_feat,
            grd_conf,
            (h, w),
        )

        # Dump visualizations if needed.
        # if visualization_dump is not None:
        #     visualization_dump["depth"] = rearrange(
        #         depths, "b (h w) s -> b h w s", h=h, w=w
        #     )
        #     visualization_dump["scales"] = rearrange(
        #         gaussians.scales, "b r spp xyz -> b (r spp) xyz"
        #     )
        #     visualization_dump["rotations"] = rearrange(
        #         gaussians.rotations, "b r spp xyzw -> b (r spp) xyzw"
        #     )

        # Optionally apply a per-pixel opacity.
        # opacity_multiplier = (
        #     rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
        #     if self.cfg.predict_opacity
        #     else 1
        # )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r spp xyz -> b (v r spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r spp i j -> b (v r spp) i j",
            ),
            rearrange(
                gaussians.opacities,
                "b v r spp -> b (v r spp)",
            ),
            rearrange(
                gaussians.color_harmonics,
                "b v r spp c d_c_sh -> b (v r spp) c d_c_sh",
            ),
            rearrange(
                gaussians.features,
                "b v r spp c -> b (v r spp) c",
            ),
            rearrange(
                gaussians.confidence,
                "b v r spp c -> b (v r spp) c",
            )
        )

    @property
    def last_layer_weights(self) -> Tensor:
        return self.to_gaussians[-1].weight
