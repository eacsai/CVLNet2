import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import data_utils as utils
import os
import torchvision.transforms.functional as TF
from lpips import LPIPS
from loss.lpips import convert_to_buffer

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from models.VGGW import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec, FeatureHead
from jacobian import grid_sample

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
# from models_ford import loss_func
from models.swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1, TransOptimizerG2SP_V2
from models.cross_attention import CrossViewAttention
import copy

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from visualize import single_features_to_RGB, sat_features_to_RGB, visualize_1d_pca, sat_features_to_RGB_2D_PCA
from gaussian.encoder import GaussianEncoder
from gaussian.decoder import GrdDecoder
from gaussian.encoder_feat import GaussianFeatEncoder
from vis_gaussian import render_projections
from models.dino_fit import DINO
from models.VGGW import L2_norm
from models.bev_net import BEVNet
from models.dpt_single import DPT
import cv2
import matplotlib.cm as cm

to_pil_image = transforms.ToPILImage()
EPS = utils.EPS

def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = depth[0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

class Model(nn.Module):
    def __init__(self, args, device=None):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.device = device

        self.level = sorted([int(item) for item in args.level.split('_')])
        self.N_iters = args.N_iters
        self.channels = [int(item) for item in self.args.channels.split('_')]
        self.gs_channels = [32, 16, 4]

        self.SatFeatureNet = VGGUnet(self.level, self.gs_channels)
        self.gaussian_encoder = GaussianEncoder(self.gs_channels[self.level[0] - 1])
        self.feat_gaussian_encoder = GaussianFeatEncoder(self.gs_channels[self.level[0] - 1])
        self.grd_decoder = GrdDecoder()
        self.lpips = LPIPS(net="vgg")

        self.dino_feat = DINO()
        self.dpt = DPT(self.dino_feat.feat_dim)
        # self.dino_feat = DinoFeat()
        convert_to_buffer(self.lpips, persistent=False)

        if self.args.share:
            self.FeatureForT = VGGUnet(self.level, self.gs_channels)
        else:
            self.GrdFeatureForT = VGGUnet(self.level, self.gs_channels)
            self.SatFeatureForT = VGGUnet(self.level, self.gs_channels)

        self.meters_per_pixel = {}
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel[level] = meter_per_pixel * (2 ** (3 - level))

        self.TransRefine = TransOptimizerG2SP_V1(self.gs_channels)

        self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
        self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

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
            self.masks[level] = mask[:, :, :, 0]

        # if self.args.use_uncertainty:
        #     self.uncertain_net = Uncertainty(self.channels)
        # self.bev_net = BEVNet()
        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.
        
    def grd_feat_projection(self, grd_img_left, grd_feat, grd_conf, grd_depth, left_camera_k, deterministic=False):
        # initial
        self.camera_k = left_camera_k.clone()
        self.camera_k[:, :1, :] = self.camera_k[:, :1, :] / grd_depth.shape[2]  # original size input into feature get network/ output of feature get network
        self.camera_k[:, 1:2, :] = self.camera_k[:, 1:2, :] / grd_depth.shape[1]
        self.camera_k = self.camera_k.unsqueeze(1)
        self.extrinsics = torch.eye(4).to(grd_img_left.device).unsqueeze(0).repeat(grd_img_left.shape[0], 1, 1).unsqueeze(1)
        # Encode the context images.
        self.gaussians = self.gaussian_encoder(
            self.grd_img_left,
            grd_feat,
            grd_conf, 
            self.camera_k, 
            self.extrinsics, 
            self.near,
            self.far, 
            deterministic
        )
        return self.gaussians
    
    def grd_pure_feat_projection(self, grd_img_left, grd_feat, grd_conf, grd_depth, left_camera_k, deterministic=False):
        # initial
        self.camera_k = left_camera_k.clone()
        self.camera_k[:, :1, :] = self.camera_k[:, :1, :] / grd_depth.shape[2]  # original size input into feature get network/ output of feature get network
        self.camera_k[:, 1:2, :] = self.camera_k[:, 1:2, :] / grd_depth.shape[1]
        self.camera_k = self.camera_k.unsqueeze(1)
        self.extrinsics = torch.eye(4).to(grd_img_left.device).unsqueeze(0).repeat(grd_img_left.shape[0], 1, 1).unsqueeze(1)
        # Encode the context images.
        self.gaussians = self.feat_gaussian_encoder(
            self.grd_img_left,
            grd_feat,
            grd_conf,
            self.camera_k, 
            self.extrinsics, 
            self.near,
            self.far, 
            deterministic
        )
        return self.gaussians

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

        camera_height = utils.get_camera_height()
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
        # return uv1

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
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, level):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat, level)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        # shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def forward_project(self, image_tensor, camera_k, depth, meter_per_pixel, sat_width=512, ori_grdH=256, ori_grdW=1024):
        origin_image_tensor = image_tensor.clone()
        B, C, grd_H, grd_W = image_tensor.shape
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
        xyz_grd = xyz_w * depth * 1.2

        # xyz_grd = xyz_grd.long()
        # xyz_grd[:,:,:,0:1] += sat_width // 2
        # xyz_grd[:,:,:,2:3] += sat_width // 2
        # B, H, W, C = xyz_grd.shape
        xyz_grd = xyz_grd.view(B*grd_H*grd_W, -1)
        xyz_grd[:, 0] = xyz_grd[:, 0] / meter_per_pixel
        xyz_grd[:, 2] = xyz_grd[:, 2] / meter_per_pixel
        xyz_grd[:, 0] = xyz_grd[:, 0].long()
        xyz_grd[:, 2] = xyz_grd[:, 2].long()

        batch_ix = torch.cat([torch.full([grd_H*grd_W, 1], ix, device=image_tensor.device) for ix in range(B)], dim=0)
        xyz_grd = torch.cat([xyz_grd, batch_ix], dim=-1)

        kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

        xyz_grd_kept = xyz_grd[kept]
        image_tensor_kept = image_tensor[kept]

        max_height = xyz_grd_kept[:,1].max()

        xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
        xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
        xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
        xyz_grd_kept = xyz_grd_kept[:,[2,0,1,3]]
        rank = torch.stack((xyz_grd_kept[:, 0] * sat_width * B + (xyz_grd_kept[:, 1] + 1) * B + xyz_grd_kept[:, 3], xyz_grd_kept[:, 2]), dim=1)
        sorts_second = torch.argsort(rank[:, 1])
        xyz_grd_kept = xyz_grd_kept[sorts_second]
        image_tensor_kept = image_tensor_kept[sorts_second]
        sorted_rank = rank[sorts_second]
        sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
        xyz_grd_kept = xyz_grd_kept[sorts_first]
        image_tensor_kept = image_tensor_kept[sorts_first]
        sorted_rank = sorted_rank[sorts_first]
        kept = torch.ones_like(sorted_rank[:, 0])
        kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]
        res_xyz = xyz_grd_kept[kept.bool()]
        res_image = image_tensor_kept[kept.bool()]
        
        # grd_image_index = torch.cat((-res_xyz[:,1:2] + grd_image_width - 1,-res_xyz[:,0:1] + grd_image_height - 1), dim=-1)
        final = torch.zeros(B,sat_width,sat_width,C).to(torch.float32).to('cuda')
        sat_height = torch.zeros(B,sat_width,sat_width,1).to(torch.float32).to('cuda')
        final[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_image

        res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
        sat_height[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_xyz[:,2].unsqueeze(-1)
        sat_height = sat_height.permute(0,3,1,2)
        # img_num = 0
        # project_grd_img = to_pil_image(final[img_num].permute(2, 0, 1))
        # project_grd_img.save('sat_feat.png')

        # project_grd_img = to_pil_image(origin_image_tensor[img_num])
        # project_grd_img.save('grd_feat.png')

        return final.permute(0,3,1,2)

    def inplane_uv(self, ori_shift_u, ori_shift_v, ori_heading, satmap_sidelength):
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength

        B = ori_heading.shape[0]
        shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
        shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
        T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]

        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)

        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - satmap_sidelength / 2

        uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]

        uv = uv_0 + satmap_sidelength / 2
        return uv

    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW, require_jac=True):
        # inputs:
        #   grd_f: ground features: B,C,H,W
        #   shift: B, S, 2
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength)  # [ sidelength,sidelength,4]

        if self.args.proj == 'geo' or self.args.proj == 'CrossAttn':
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
            # [B, H, W, 2], [B, H, W, 1]

        grd_f_trans, new_jac = grid_sample(grd_f, uv, None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        grd_f_trans = grd_f_trans * mask[:, None, :, :, 0]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
            grd_c_trans = grd_c_trans * mask[:, None, :, :, 0]
        else:
            grd_c_trans = None


        return grd_f_trans, grd_c_trans, uv, mask

    def NeuralOptimizer(self, grd_feat_dict, sat_feat_dict, B, left_camera_k=None, ori_grdH=None, ori_grdW=None):

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in self.level:
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]

                if self.args.stage == 4 or self.args.stage == 3:
                    A = sat_feat.shape[-1]
                    overhead_feat, _, _, _ = self.project_grd_to_map(
                        grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)
                else:
                    uv = self.inplane_uv(shift_u, shift_v, heading, sat_feat.shape[-1])
                    overhead_feat, _ = grid_sample(
                        grd_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                        uv, jac=None)
                # elif self.args.stage == 1:
                #     A = sat_feat.shape[-1]
                #     overhead_feat, _, _, _ = self.project_grd_to_map(
                #         grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                shift_u_new, shift_v_new, heading_new = self.Trans_update(
                    shift_u, shift_v, heading, overhead_feat, sat_feat, level)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        return shift_lats, shift_lons, thetas

    def forward(self, sat_align_cam, sat_map, grd_img_left, grd_depth, left_camera_k, gt_heading=None, gt_shift_u=None, gt_shift_v=None, train=False, loop=None, save_dir=None):
        '''
        rot_corr
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            grd_depth: [B, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')
        # sat_align_cam_ = transforms.ToPILImage()(sat_align_cam[0])
        # sat_align_cam_.save('sat_align_cam.png')
        #
        # uv = self.inplane_uv(gt_shift_u, gt_shift_v, gt_heading, sat_map.shape[-1])
        # sat_align_cam_trans, _ = grid_sample(
        #     sat_align_cam,
        #     uv, jac=None)
        # sat_align_cam_trans = transforms.ToPILImage()(sat_align_cam_trans[0])
        # sat_align_cam_trans.save('sat_align_cam_trans.png')


        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        self.near = torch.ones(B, 1).to(grd_img_left.device) * 0.5
        self.far = torch.ones(B, 1).to(grd_img_left.device) * 100
        self.grd_img_left = F.interpolate(grd_img_left, size=(64, 256), mode='bilinear', align_corners=False)
        # self.grd_img_left = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
        self.sat_map = F.interpolate(sat_map, size=(128, 128), mode='bilinear', align_corners=False)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        if self.args.stage == 0:
            sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat_map)
            over_feat_dict, over_conf_dict = self.SatFeatureNet(sat_align_cam)
            # not sure whether mask should be appliced at image level or feature level

            shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B)

            for _, level in enumerate(self.level):
                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict[level]
                over_feat = over_feat_dict[level]
                over_conf = over_conf_dict[level]

                A = sat_feat.shape[-1]
                uv = self.inplane_uv(shift_u, shift_v, gt_heading, A)
                overhead_feat, _ = grid_sample(
                    over_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None)
                overhead_conf, _ = grid_sample(
                    over_conf * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None
                )

                crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(overhead_feat, [crop_H, crop_W])
                overhead_conf = TF.center_crop(overhead_conf, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = overhead_conf

            return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, self.masks, shift_lats, shift_lons, thetas, None

        elif self.args.stage == 1:
            decoder_grd = self.grd_projection(grd_img_left, grd_depth, left_camera_k)
            grd_color = decoder_grd.color.clip(min=0, max=1) 
            test_img = to_pil_image(grd_color[0])
            test_img.save(f'grd_weak_maskfov1.png')
            _, _, H, W = grd_color.shape
            grd_depth_imgs = F.interpolate(grd_depth.unsqueeze(1), (H, W), mode='bilinear', align_corners=True).squeeze(1)
            grd_left_imgs = F.interpolate(grd_img_left, (H, W), mode='bilinear', align_corners=True)
            depth_l1_loss = F.l1_loss(decoder_grd.depth, grd_depth_imgs, reduction='mean')
            rgb_mse_loss = F.mse_loss(decoder_grd.color, grd_left_imgs, reduction='mean')
            lpips_loss = self.lpips.forward(grd_color, grd_left_imgs, normalize=True).mean()
            render_loss = depth_l1_loss + rgb_mse_loss * 20 + lpips_loss
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                grd2sat_gaussian_color1, grd2sat_gaussian_feat1 = render_projections(self.gaussians, (512,512), extra_label='prob')
                
                sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd2sat_gaussian_color1)

                if self.args.rotation_range > 0:
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            grd2sat_gaussian_color2, grd2sat_gaussian_feat2 = render_projections(self.gaussians, (128,128), extra_label='prob', heading=heading)
            test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov1.png')

            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_img_left, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')

            # test_img = to_pil_image(sat_map[0].clip(min=0, max=1))
            # test_img.save(f'origin_sat.png')

            if self.args.share:
                grd_feat_dict_forT, grd_conf_dict_forT = L2_norm(grd2sat_gaussian_feat2), torch.ones_like(grd2sat_gaussian_feat2, device=grd2sat_gaussian_feat2.device)
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd2sat_gaussian_color)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)

            grd_uv_dict = {}
            mask_dict = {}
            level = 1
            # self.level = level
            sat_feat = sat_feat_dict_forT[level]
            satmap_sidelength = sat_feat.shape[-1]
            XYZ_1 = self.sat2world(satmap_sidelength)                
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, left_camera_k, ori_grdH, ori_grdW,
                                                            ori_grdH, ori_grdW)
            g2s_feat_dict[level] = grd_feat_dict_forT * mask.permute(0, 3, 1, 2)
            g2s_conf_dict[level] = grd_conf_dict_forT * mask.permute(0, 3, 1, 2)
            grd_uv_dict[level] = uv
            mask_dict[level] = mask

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss

        elif self.args.stage == 2:

            level = self.level[0]
            grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
            self.grd_img_left = self.grd_img_left.unsqueeze(1)

            grd_gaussian = self.grd_feat_projection(self.grd_img_left, None, None, grd_depth, left_camera_k)
            
            decoder_grd = self.grd_decoder.forward(
                grd_gaussian,     # Sample from variational Gaussians
                self.extrinsics,
                self.camera_k,
                self.near,
                self.far,
                (64, 256),
            )

            grd2sat_gaussian_color, _, _ = render_projections(grd_gaussian, (512,512))
            test_img = to_pil_image(grd2sat_gaussian_color[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_ori_128.png')
            test_img = to_pil_image(self.grd_img_left[0,0].clip(min=0, max=1))
            test_img.save(f'grd_ori_128.png')
            test_img = to_pil_image(decoder_grd.color[0,0].clip(min=0, max=1))
            test_img.save(f'grd_weak_ori_128.png')
            grd_depth = F.interpolate(grd_depth.unsqueeze(1), (64, 256), mode='bilinear', align_corners=True).squeeze(1)
            onlyDepth(grd_depth, 'grd_depth_ori_128.png')
            onlyDepth(decoder_grd.depth[0], 'grd_depth_128.png')
            with torch.no_grad():
                if self.args.rotation_range > 0:
                    sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat_map)
                    over_feat_dict, over_conf_dict = self.SatFeatureNet(grd2sat_gaussian_color)
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B)
                else:
                    thetas = torch.zeros([B, 1, 1], dtype=torch.float32, requires_grad=False, device=sat_map.device)
            
            return None, None, decoder_grd, None, None, None, None, thetas, None
        

        elif self.args.stage == 3:
            # sat_feat_dict = {}
            # sat_conf_dict = {}
            level = self.level[0]
            # if self.args.share:
            #     sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
            #     grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
            # else:
            #     grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd_img_left)
            #     sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)
            self.gaussian_encoder.eval()
            with torch.no_grad():
                # dino
                sat_feat = self.dino_feat(sat_map)
                grd_feat = self.dino_feat(grd_img_left)
                if isinstance(sat_feat, (tuple, list)):
                    sat_feats = [_f.detach() for _f in sat_feat]
                if isinstance(grd_feat, (tuple, list)):
                    grd_feats = [_f.detach() for _f in grd_feat]
            
            # TODO: use two dpt and upsample grd_feat
            # dpt
            sat_feat, sat_conf = self.dpt(sat_feats)
            grd_feat, grd_conf = self.dpt(grd_feats)

            sat_feat_dict_forT = {}
            sat_conf_dict_forT = {}
            grd_feat_dict_forT = {}
            grd_conf_dict_forT = {}

            sat_feat_dict_forT[level] = sat_feat
            sat_conf_dict_forT[level] = sat_conf
            grd_feat_dict_forT[level] = grd_feat
            grd_conf_dict_forT[level] = grd_conf
            # dpt over

            self.grd_img_left = self.grd_img_left.unsqueeze(1)
            
            grd_gaussian = self.grd_feat_projection(
                self.grd_img_left, 
                grd_feat_dict_forT[level].unsqueeze(1), 
                grd_conf_dict_forT[level].unsqueeze(1), 
                grd_depth, 
                left_camera_k
            )
            
            render_loss = torch.tensor(0.0).to(self.grd_img_left.device)
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                if self.args.rotation_range > 0:
                    # grd2sat_gaussian_color1, grd2sat_gaussian_feat1, grd2sat_gaussian_conf1 = render_projections(grd_gaussian, (512,512))
                
                    sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                    grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd_img_left)
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    thetas = heading.unsqueeze(1)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            grd2sat_gaussian_color2, grd2sat_gaussian_feat2, grd2sat_gaussian_conf2 = render_projections(grd_gaussian, (128,128), heading=heading, rot_range=self.args.rotation_range)
            test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
            test_img.save(f'sat_weak_maskfov3.png')
            # single_features_to_RGB(grd2sat_gaussian_feat2)
            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_feat, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # sat_features_to_RGB_2D_PCA(grd2sat_gaussian_feat2, grd_origin)

            
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')
            test_img = to_pil_image(grd_img_left[0].clip(min=0, max=1))
            test_img.save(f'grd.png')
            test_img = to_pil_image(self.sat_map[0].clip(min=0, max=1))
            test_img.save(f'origin_sat.png')

            # sat_feat = sat_feat_dict_forT[level]
            # grd2sat_forward = self.forward_project( grd_img_left, left_camera_k, grd_depth, self.meters_per_pixel[1], 128, ori_grdH, ori_grdW)
            # test_img = to_pil_image(grd2sat_forward[0].clip(min=0, max=1))
            # test_img.save(f'forward.png')            

            mask_dict = {}
            mask = (grd2sat_gaussian_feat2 != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
            mask_dict[level] = mask

            g2s_feat_dict[level] = grd2sat_gaussian_feat2
            g2s_conf_dict[level] = grd2sat_gaussian_conf2

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])
                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])
                g2s_feat_dict[level] = g2s_feat
                # g2s_conf_dict[level] = (g2s_feat != 0).any(dim=1, keepdim=True).float()
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss

        # TODO: stage 4
        elif self.args.stage == 4:
            # sat_feat_dict = {}
            # sat_conf_dict = {}            
            level = self.level[0]
            with torch.no_grad():
                # dino
                sat_feat = self.dino_feat(sat_map)
                grd_feat = self.dino_feat(grd_img_left)
                if isinstance(sat_feat, (tuple, list)):
                    sat_feats = [_f.detach() for _f in sat_feat]
                if isinstance(grd_feat, (tuple, list)):
                    grd_feats = [_f.detach() for _f in grd_feat]
            
                # TODO: use two dpt and upsample grd_feat
                # dpt
                sat_feat, sat_conf = self.dpt(sat_feats)
                grd_feat, grd_conf = self.dpt(grd_feats)
            

            sat_feat_dict_forT = {}
            sat_conf_dict_forT = {}
            grd_feat_dict_forT = {}
            grd_conf_dict_forT = {}

            sat_feat_dict_forT[level] = sat_feat
            sat_conf_dict_forT[level] = sat_conf
            grd_feat_dict_forT[level] = grd_feat
            grd_conf_dict_forT[level] = grd_conf
            # dpt over

            self.grd_img_left = self.grd_img_left.unsqueeze(1)
            grd_gaussian = self.grd_pure_feat_projection(
                self.grd_img_left, 
                grd_feat.unsqueeze(1),
                grd_conf.unsqueeze(1),
                grd_depth, 
                left_camera_k
            )
            
            decoder_grd = self.grd_decoder.forward(
                grd_gaussian,     # Sample from variational Gaussians
                self.extrinsics,
                self.camera_k,
                self.near,
                self.far,
                (64, 256),
            )
            
            grd_color = decoder_grd.color.squeeze(1)

            grd_depth_imgs = F.interpolate(grd_depth.unsqueeze(1), (64, 256), mode='bilinear', align_corners=True)
            # grd_left_imgs = self.grd_img_left.squeeze(1)
            depth_l1_loss = F.l1_loss(decoder_grd.depth, grd_depth_imgs, reduction='mean')
            conf_l1_loss = F.l1_loss(decoder_grd.confidence.squeeze(1), grd_conf, reduction='mean')
            # rgb_mse_loss = F.mse_loss(grd_color, grd_left_imgs, reduction='mean')
            # lpips_loss = self.lpips.forward(grd_color, grd_left_imgs, normalize=True).mean()
            feat_l1_loss = F.l1_loss(grd_feat, decoder_grd.feature.squeeze(1), reduction='mean')
            render_loss = feat_l1_loss * 100 + depth_l1_loss + conf_l1_loss
            # render_loss = rgb_mse_loss * 20 + lpips_loss
            # render_loss = depth_l1_loss * 0.01 + rgb_mse_loss + lpips_loss * 0.05
            # render_loss = torch.tensor(0.0).to(self.grd_img_left.device)
            # ----------------- Rotation Stage ---------------------------
            with torch.no_grad():
                if self.args.rotation_range > 0:
                    # grd2sat_gaussian_color1, grd2sat_gaussian_feat1, grd2sat_gaussian_conf1 = render_projections(grd_gaussian, (512,512))
                
                    sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                    grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd_img_left)
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    thetas = heading.unsqueeze(1)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            _, grd2sat_gaussian_feat2, grd2sat_gaussian_conf2 = render_projections(grd_gaussian, (128,128), heading=heading)

            # single_features_to_RGB(grd2sat_gaussian_feat2, img_name = 'sat_weak_maskfov4_feat.png')
            # grd_origin, _, _, _ = self.project_grd_to_map(
            #         grd_img_left, None, shift_u, shift_v, heading, left_camera_k, 512, ori_grdH,
            #         ori_grdW,
            #         require_jac=False)
            # test_img = to_pil_image(grd_origin[0].clip(min=0, max=1))
            # test_img.save(f'grd_origin.png')

            # test_img = to_pil_image(sat_map[0].clip(min=0, max=1))
            # test_img.save(f'origin_sat.png')

            mask_dict = {}
            sat_feat = sat_feat_dict_forT[level]
            satmap_sidelength = sat_feat.shape[-1]
            # XYZ_1 = self.sat2world(satmap_sidelength)                
            # uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, left_camera_k, ori_grdH, ori_grdW,
            #                                                 ori_grdH, ori_grdW)
                        
            mask = (grd2sat_gaussian_feat2 != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
            # mask = (grd2sat_gaussian_feat2 != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
            mask_dict[level] = mask

            for _, level in enumerate(self.level):

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(grd2sat_gaussian_feat2, [crop_H, crop_W])
                g2s_conf = TF.center_crop(grd2sat_gaussian_conf2, [crop_H, crop_W])
                # g2s_conf = (g2s_feat != 0).any(dim=1, keepdim=True).float()
                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss

def batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, masks=None):
    '''
    compute corr_maps for training
    result corr_map has a shape of [M, N, H, W],
    M is the number of satellite images and N is the number of ground images
    '''

    levels = sorted([int(item) for item in args.level.split('_')])
    corr_maps = {}
    for _, level in enumerate(levels):
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape


        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)   # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(0, B):
                    denom_grd = torch.sum(F.conv2d(sat_conf_pow[i:i+1, :, :, :].repeat(1, C, 1, 1), g2s_feat_conf_pow), dim=1)
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

                # corr = corr / denominator_sat / denominator_grd

            else:

                # numerator
                signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_pow = (sat_feat).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

                # corr = corr / denominator_sat / denominator_grd

        else:
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()

            signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            # fixme: denominator
            # denominator_sat1 = []
            # mask_kernel = TF.center_crop(masks[level], [crop_H, crop_W]).float().unsqueeze(1).repeat(B, 1, 1, 1)
            # for i in range(0, B):
            #     denom_sat = torch.sum(F.conv2d(sat_feat.pow(2)[i, :, None, :, :], mask_kernel), dim=0)
            #     denominator_sat1.append(denom_sat)
            # denominator_sat1 = torch.sqrt(torch.stack(denominator_sat1, dim=0))  # [B (M), B (N), H, W]
            
            l2_norm_kernel = mask.repeat(1, C, 1, 1)
            sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)
            denominator_sat = torch.sqrt(sat_feat_squared_sum + 1e-8)
            # single_features_to_RGB(g2s_feat)
            # single_features_to_RGB(g2s_feat * mask)
            # original
            # denominator_sat_ori = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            # denominator_sat_ori = torch.sqrt(torch.sum(denominator_sat_ori, dim=1, keepdim=True))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = 2 - 2 * corr / denominator  # [B, B, H, W]

        corr_maps[level] = corr

    return corr_maps


def weak_supervise_loss(corr_maps):
    '''
    triplet loss/ metric learning loss for self-supervision
    corr_maps: dict
    key -- level; value -- corr map
    '''
    losses = []
    for key, corr in corr_maps:
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meter_per_pixels, GPS_error=5):
    '''
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []
    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
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
        meter_per_pixel = meter_per_pixels[level]
        w = (torch.round(W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)).long() # [B]
        h = (torch.round(H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)).long() # [B]
        radius = int(np.ceil(GPS_error / meter_per_pixel))
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius)
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius)
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius)
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius)
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))


def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meters_per_pixel):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    # cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    # sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    #
    # gt_delta_x = gt_shift_u[:, 0] * args.shift_range_lon
    # gt_delta_y = gt_shift_v[:, 0] * args.shift_range_lat
    #
    # gt_delta_x_rot = - gt_delta_x * cos - gt_delta_y * sin
    # gt_delta_y_rot = gt_delta_x * sin - gt_delta_y * cos

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        meter_per_pixel = meters_per_pixel[level]

        w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
        h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, meter_per_pixels, gt_heading, masks=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])
    meter_per_pixel = meter_per_pixels[level]

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

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

        mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
        l2_norm_kernel = mask.repeat(1, C, 1, 1)
        sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)[0]
        denominator_sat = torch.maximum(torch.sqrt(sat_feat_squared_sum + 1e-8), torch.ones_like(sat_feat_squared_sum) * 1e-6)  # 滑动窗口的 L2 范数
        # denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        # denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))
        
        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator  # [B, H, W]

    corr_H = int(args.shift_range_lat * 3 / meter_per_pixel)
    corr_W = int(args.shift_range_lon * 3 / meter_per_pixel)

    corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)

    if args.visualize:
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * np.power(2, 3 - level)
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * np.power(2, 3 - level)
        return pred_u, pred_v, corr

    else:

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

        pred_u1 = pred_u * cos + pred_v * sin
        pred_v1 = - pred_u * sin + pred_v * cos

        return pred_u1, pred_v1, corr



def corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args,
                                              sat_uncer_dict=None):
    levels = [int(item) for item in args.level.split('_')]

    corr_maps = {}
    for level in levels:

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
                signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
                denominator_grd = torch.sqrt(denominator_grd)

            else:

                # numerator
                signal = sat_feat.reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        # if args.use_uncertainty:
        #     denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = corr / denominator

        corr_maps[level] = 2 - 2 * corr

    return corr_maps




def loss_func(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0, 0] - shift_lat_delta[-1, -1]  # scalar
    shift_lon_decrease = shift_lon_delta[0, 0] - shift_lon_delta[-1, -1]  # scalar
    thetas_decrease = thetas_delta[0, 0] - thetas_delta[-1, -1]  # scalar

    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
    loss_decrease = losses[0, 0] - losses[-1, -1]  # scalar
    loss = torch.mean(losses)  # mean or sum
    loss_last = losses[-1]

    return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
        shift_lat_delta[-1, -1], shift_lon_delta[-1, -1], thetas_delta[-1, -1]

