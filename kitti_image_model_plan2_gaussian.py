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
    def __init__(self, args, device, mode = 'gaussian'):  # device='cuda:0',
        super(Model, self).__init__()
        self.device = device
        self.args = args
        self.d_color_sh = 25
        self.d_feature_sh = 9
        self.n_feature_channels = 8
        self.variational = "gaussians"
        generator_lr = 1.5e-4
        autoencoder_lr = 4.5e-6 * args.batch_size
        discriminator_lr = 4.5e-6 * args.batch_size
        local_lr = 1e-4
        self.supersampling_factor = 8
        self.save_path = './ModelsGaussian/' + args.name
        self.level = args.level

        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        self.gaussian_encoder = GaussianEncoder()
        self.grd_decoder = GrdDecoder()
        self.autoencoder = AutoencoderKL()
        # self.sat_encoder = AutoencoderKL()
        self.sat_encoder = VGGUnet(self.level)
        self.grd_encoder = VGGUnet(self.level)
        self.discriminator = DiscriminatorPatchGan()

        self.lpips = LPIPS(net="vgg")
        self.local_loss = LocalLoss(args.shift_range_lat, args.shift_range_lon, args.rotation_range)
        convert_to_buffer(self.lpips, persistent=False)

        self.generator_optimizer = optim.Adam([
            {
                "params": self.autoencoder.parameters(),  # 第一个参数组
                "lr": autoencoder_lr,
                "betas": (0.5, 0.9),
            },
            {
                "params": chain(self.gaussian_encoder.parameters(), self.grd_decoder.parameters()),  # 第二个参数组
                "lr": generator_lr,
                "betas": (0.9, 0.999)
            }
        ])

        self.generator_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.generator_optimizer,
            start_factor=1 / args.warm_up_steps,
            end_factor=1.0,
            total_iters=args.warm_up_steps,
        )

        self.discriminator_optimizer = optim.Adam(
            params= self.discriminator.parameters(),  # 参数组
            lr=discriminator_lr,              
            eps=1e-08,            # 精度参数
            weight_decay=0,       # 权重衰减
            amsgrad=False         # 是否使用 AMSGrad 变体
        )

        self.local_optimizer = optim.Adam(
            params= chain(self.sat_encoder.parameters(), self.local_loss.parameters(), self.grd_encoder.parameters()),  # 参数组
            lr=local_lr,
        )

        if mode=='gaussian':
            self.global_step = 0
            self.save_model_names = self.load_model_names = ['gaussian_encoder', 'grd_decoder', 'autoencoder', 'discriminator']
        elif mode=='local':
            self.global_step = 1000000
            self.load_model_names = ['gaussian_encoder', 'grd_decoder', 'autoencoder']
            self.save_model_names = ['gaussian_encoder_loc', 'grd_decoder_loc', 'autoencoder_loc', 'local_loss', 'sat_encoder', "grd_encoder"]
            self.load_networks('5')
            self.set_requires_grad(self.discriminator, False)

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
            self.gaussians.sample() if self.variational in ("gaussians", "none") else self.gaussians.flatten(),     # Sample from variational Gaussians
            self.extrinsics,
            self.camera_k,
            near,
            far,
            (128, 512),
            return_colors=True,
            return_features=self.global_step >= L1_step,
        )
        return decoder_out

    def sat_prjection(self):
        color, latent = render_projections(self.gaussians.sample(), (512,512), extra_label='prob')
        z = self.rescale(latent, Fraction(1, self.supersampling_factor))
        skip_z = torch.cat((color, latent), dim=-3)
        if self.global_step >= sat_prjection_step:
            dec = self.autoencoder.decode(z, skip_z)
            sat_img = dec
        else:
            dec = None
            sat_img = color

        if self.global_step % 10 == 0:
            projection_img = to_pil_image(sat_img[0].clip(min=0, max=1))
            projection_img.save(f"sat_projecton_vae.png")
        return dec

    def gaussian_generator(self, out: DecoderOutput, use_generator=True):
        # color.register_hook(print_grad)   
        latent = out.feature_posterior.sample()
        z = self.rescale(latent, Fraction(1, self.supersampling_factor))
        skip_z = torch.cat((out.color, latent), dim=-3)
        if self.global_step >= L1_step and use_generator:
            dec, _ = self.autoencoder.decode(z, skip_z)
        else:
            dec = None
        self.refine_img = dec
        # render_loss    
        rgb_mse_loss = F.mse_loss(out.color, self.grd_img_left, reduction='mean')
        depth_l1_loss = F.l1_loss(out.depth.unsqueeze(-1), self.grd_depth, reduction='mean')
        if self.global_step >= raw_Lpips_step:
            raw_lpips_loss = self.lpips.forward(out.color, self.grd_img_left,  normalize=True)
        else:
            raw_lpips_loss = torch.tensor(0, dtype=torch.float32, device=out.color.device)
        
        self.render_loss = rgb_mse_loss * 10 + depth_l1_loss + raw_lpips_loss.mean() * 0.5
        # refine_loss
        if self.global_step >= L1_step and use_generator:
            refine_l1_loss = F.l1_loss(self.refine_img, self.grd_img_left, reduction='mean')
        else:
            refine_l1_loss = torch.tensor(0, dtype=torch.float32, device=out.color.device)
        if self.global_step >= refine_Lpips_step and use_generator:
            refine_lpips_loss = self.lpips.forward(self.refine_img, self.grd_img_left,  normalize=True)
        else:
            refine_lpips_loss = torch.tensor(0, dtype=torch.float32, device=out.color.device)

        self.refine_loss = refine_l1_loss + refine_lpips_loss.mean()
        # generator_loss
        if self.global_step >= discriminator_loss_active_step and use_generator:
            logits_fake = self.discriminator(self.refine_img)
            generator_loss = -logits_fake.mean()
            last_layer_weights = self.last_layer_weight()
            adaptive_weight = self.get_adaptive_weight(self.refine_los, generator_loss, last_layer_weights) * 0.5
        else:
            generator_loss = torch.tensor(0, dtype=torch.float32, device=out.color.device)
            adaptive_weight = torch.tensor(0, dtype=torch.float32, device=out.color.device)
        
        loss = self.render_loss + self.refine_loss + generator_loss * adaptive_weight
        
        if dec is not None:
            test_img = to_pil_image(dec[0].clip(min=0, max=1))
        else:
            test_img = to_pil_image(out.color[0].clip(min=0, max=1))
        test_img.save('probabilistic_vae.png')
        # test_img = to_pil_image(self.grd_img_left[0])
        # test_img.save('gt.png')
        return loss

    def forward(self, sat_map, grd_img_left, project_map, grd_depth, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='local'):
        if mode == 'gaussian':
            # train the grd generator
            self.set_requires_grad(self.discriminator, False)
            self.generator_optimizer.zero_grad()

            decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
            generator_loss = self.gaussian_generator(decoder_grd)
            generator_loss.backward()
            # optimize the generator
            for param_group in self.generator_optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=0.5)
            self.generator_optimizer.step()
            self.generator_scheduler.step()

            # train the discriminator
            discriminator_loss = 0
            if self.global_step >= discriminator_loss_active_step:
                self.set_requires_grad(self.discriminator, True)
                self.discriminator_optimizer.zero_grad()
                logits_fake = self.discriminator(self.refine_img.detach())
                logits_real = self.discriminator(self.grd_img_left)
                loss_fake = F.relu(1 + logits_fake).mean()
                loss_real = F.relu(1 - logits_real).mean()  # NOTE negative
                discriminator_loss = loss_fake * 0.5 + loss_real * 0.5
                discriminator_loss.backward()

                # optimize the discriminator
                self.discriminator_optimizer.step()

            # train the sat generator
            self.sat_prjection()

            self.global_step = self.global_step + sat_map.shape[0]
            self.loss = generator_loss + discriminator_loss
            return self.loss
        
        elif mode == 'local':
            self.local_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()

            decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
            # render_loss = self.gaussian_generator(decoder_grd, use_generator=False)
            
            grd_map = self.sat_prjection()
            grd_feat_list, grd_conf_list = self.grd_encoder(grd_map)
            grd_feat = grd_feat_list[-1]
            sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
            sat_feat = sat_feat_list[-1]
            local_loss = self.local_loss(grd_feat, sat_feat, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            # total_loss = render_loss + local_loss
            total_loss = local_loss
            total_loss.backward()

            # optimize the generator
            for param_group in self.generator_optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=0.5)
            self.generator_optimizer.step()
            # optimize the local loss
            self.local_optimizer.step()
            self.global_step = self.global_step + sat_map.shape[0]
            self.loss = local_loss
            return self.loss
        
        elif mode == 'test':
            self.grd_projection(grd_img_left, grd_depth, left_camera_k)
            grd_map = self.sat_prjection()
            grd_feat_list, grd_conf_list = self.grd_encoder(grd_map)
            grd_feat = grd_feat_list[-1]
            sat_feat_list, sat_conf_list = self.sat_encoder(sat_map)
            sat_feat = sat_feat_list[-1]
            pred_u1, pred_v1 = self.local_loss(grd_feat, sat_feat, gt_shift_u, gt_shift_v, gt_heading, mode='test')
            return pred_u1, pred_v1
    
    def save_networks(self, epoch):
        for name in self.save_model_names:
            save_filename = f'{name}_{epoch}.pth'
            save_path = os.path.join(self.save_path, save_filename)
            net = getattr(self, name.replace('_loc', ''))
            torch.save(net.cpu().state_dict(), save_path)
            net.to(self.device)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.load_model_names:
            if isinstance(name, str):
                load_filename = f'{name}_{which_epoch}.pth'
                load_path = os.path.join(self.save_path, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

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

    def last_layer_weight(self) -> Tensor:
        res = self.autoencoder.last_layer_weights
        if res is None:
            res = self.grd_decoder.last_layer_weights
            if res is None:
                res = self.gaussian_encoder.last_layer_weights
                if res is None:
                    raise ValueError("Could not find last layer weights in autoencoder, decoder, or encoder")
        return res
    
    def get_adaptive_weight(
        self,
        nll_loss: Float[Tensor, ""], 
        g_loss: Float[Tensor, ""], 
        last_layer_weights: Tensor
    ):
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weights, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weights, retain_graph=True)[0]

        weight = torch.linalg.norm(nll_grads) / (torch.linalg.norm(g_grads) + 1e-4)
        weight = torch.clamp(weight, 0.0, 1.0).detach()
        return weight