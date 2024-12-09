import torch
from torch import Tensor, nn

import data_utils
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class LocalLoss(nn.Module):
    def __init__(self, shift_range_lat, shift_range_lon, rotation_range):
        super(LocalLoss, self).__init__()

        self.meters_per_pixel = []
        meter_per_pixel = data_utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))
        self.shift_range_lat = shift_range_lat
        self.shift_range_lon = shift_range_lon
        self.rotation_range = rotation_range

    def forward(self, grd_feat, sat_feat, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        corr_maps = []
        meter_per_pixel = self.meters_per_pixel[-2]
        B, _, A, _ = sat_feat.shape
        crop_H = int(A - self.shift_range_lat * 3 / meter_per_pixel)
        crop_W = int(A - self.shift_range_lon * 3 / meter_per_pixel)
        g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
        g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

        s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

        denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
        denominator = torch.sum(denominator, dim=1)  # [B, H, W]
        denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
        corr = 2 - 2 * corr / denominator

        B, corr_H, corr_W = corr.shape

        corr_maps.append(corr)

        max_index = torch.argmin(corr.reshape(B, -1), dim=1)
        pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

        cos = torch.cos(gt_heading[:, 0] * self.rotation_range / 180 * torch.pi)
        sin = torch.sin(gt_heading[:, 0] * self.rotation_range / 180 * torch.pi)

        pred_u1 = pred_u * cos + pred_v * sin
        pred_v1 = - pred_u * sin + pred_v * cos

        local_loss = self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        if mode == 'train':
            return local_loss
        else:
            return pred_u1, pred_v1 
    
    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.rotation_range / 180 * torch.pi)
        sin = torch.sin(gt_heading[:, 0] * self.rotation_range / 180 * torch.pi)

        gt_delta_x = - gt_shift_u[:, 0] * self.shift_range_lon
        gt_delta_y = - gt_shift_v[:, 0] * self.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[-2]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))