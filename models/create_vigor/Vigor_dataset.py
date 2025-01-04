import random

import numpy as np
import os
from PIL import Image
import PIL
from torch.utils.data import Dataset, Subset

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import torch.nn.functional as F


num_thread_workers = 8
root = '/data/dataset/wqw/VIGOR'


model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

depth_v2_load_from = "/home/wangqw/video_program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"

class VIGORDataset(Dataset):
    def __init__(self, root, rotation_range, label_root='splits__corrected', split='same', train=True, transform=None, pos_only=True, amount=1.):
        self.root = root
        self.rotation_range = rotation_range
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only

        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]

        if self.split == 'same':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'cross':
            if self.train:
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']

        self.meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                                     'Seattle': 0.100817 * 640 / 512,
                                     'SanFrancisco': 0.118141 * 640 / 512,
                                     'Chicago': 0.111262 * 640 / 512}

        # load sat list
        self.sat_list = []
        self.sat_index_dict = {}

        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', sat_list_fname, idx)
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print('Sat loaded, data size:{}'.format(self.sat_data_size))

        # load grd list
        self.grd_list = []
        # self.grd_params = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == 'same':
                if self.train:
                    label_fname = os.path.join(self.root, self.label_root, city, 'same_area_balanced_train__corrected.txt')
                else:
                    label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test__corrected.txt')
            elif self.split == 'cross':
                label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced__corrected.txt')

            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, 'pano_mask_sky', data[0]))
                    # self.grd_params.append(os.path.join(self.root, city, 'pers_imgs', data[0].replace('.jpg', '_pers.pt')))
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', label_fname, idx)

        self.data_size = int(len(self.grd_list) * amount)
        self.grd_list = self.grd_list[: self.data_size]
        self.label = self.label[: self.data_size]
        self.delta = self.delta[: self.data_size]
        print('Grd loaded, data size:{}'.format(self.data_size))
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

        # load depth model
        # depth_anything_v2 = DepthAnythingV2(**{**model_configs["vitl"], "max_depth": 80})
        # depth_anything_v2.load_state_dict(torch.load(depth_v2_load_from, map_location="cpu"))
        # self.depth_anything_v2 = depth_anything_v2.to("cuda").eval()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')         
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (320, 640))  # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)
        # generate a random rotation
        rotation = np.random.uniform(low=-1.0, high=1.0)  #
        rotation_angle = rotation * self.rotation_range
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation_angle / 180) * grd.size()[2] / 2).int()).item(),
                         dims=2)

        # satellite
        pos_index = 0
        sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
        [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]

        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        row_offset = np.round(row_offset / height_raw * height)
        col_offset = np.round(col_offset / width_raw * width)

        # groundtruth location on the aerial image
        gt_shift_y = row_offset / height * 4  # -L/4 ~ L/4  -1 ~ 1
        gt_shift_x = -col_offset / width * 4  #

        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'

        # grd_params = torch.load(self.grd_params[idx])
        # depth_imgs = grd_params['depth_imgs']
        # pers_imgs = grd_params['pers_imgs']
        # camera_k = grd_params['camera_k']
        # extrinsics = grd_params['extrinsics']
        pers_path = '/data/dataset/wqw/VIGOR/' + city + '/6_pers_imgs_160_new'
        if not os.path.exists(pers_path):
            os.makedirs(pers_path)
        
        grd_name = self.grd_list[idx].split('/')[-1].replace('.jpg', '')
        save_path = os.path.join(pers_path, f'{grd_name}_pers.pt')

        # cerate dataset
        return grd, save_path


def load_vigor_data(batch_size, area="same", rotation_range=0, train=True, weak_supervise=True, amount=1.):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    transform_grd = transforms.Compose([
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    transform_sat = transforms.Compose([
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    vigor = VIGORDataset(root, rotation_range, split=area, train=train, transform=(transform_grd, transform_sat),
                         amount=amount)

    train_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_dataloader
