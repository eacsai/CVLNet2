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

# from twenty_pano_utils import *
from six_pano_utils import *
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

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        self.global_step = 0

    def create_data(self, grd, save_path):

        with torch.no_grad():
            pers_imgs, extrinsics, camera_k = split_panorama(grd, gen_res=160, device=self.device)
            b,v,c,h,w = pers_imgs.shape
            camera_k = camera_k.unsqueeze(0).repeat(b, v, 1, 1)
            extrinsics = extrinsics.unsqueeze(0).repeat(b, 1, 1, 1)
            
            depth = self.depth_anything_v2.infer_image(pers_imgs.reshape(b*v,c,h,w), 518)
            depth = depth.reshape(b,v,h,w)
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
