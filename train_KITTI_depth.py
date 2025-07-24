import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
from dataLoader.KITTI_dataset_Metirc3d import load_train_data, load_test1_data, load_test2_data
import scipy.io as scio
from torchvision import transforms
import ssl
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.cm as cm # 导入 colormap 模块
import matplotlib.colors as mcolors # 导入 colors 模块

to_pil_img = transforms.ToPILImage()
ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

# from models_ford import loss_func, loss_func_l2
from models.models_kitti_nips import Model, batch_wise_cross_corr, corr_for_translation, weak_supervise_loss, \
    Weakly_supervised_loss_w_GPS_error, corr_for_accurate_translation_supervision, GT_triplet_loss, loss_func

import numpy as np
import os
import argparse
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = depth[0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

def test1(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))

    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for item in Data[:9]]
            
            pred_depth, confidence, output_dict = net.inference({'input': grd_left_imgs_ori})
            pred_depth = F.interpolate(pred_depth, size=(grd_left_imgs_ori.shape[2], grd_left_imgs_ori.shape[3]), mode='bilinear', align_corners=False)
            mask = (grd_left_imgs != 0).any(dim=1, keepdim=True).float()
            pred_depth = pred_depth * mask

            depth_paths = Data[-1]
            for j in range(len(depth_paths)):
                depth = pred_depth[j,0]
                depth = depth / depth.max() * 80
                torch.save(depth, depth_paths[j])

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)

def test2(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))
    
    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for item in Data[:9]]
            
            pred_depth, confidence, output_dict = net.inference({'input': grd_left_imgs_ori})
            pred_depth = F.interpolate(pred_depth, size=(grd_left_imgs_ori.shape[2], grd_left_imgs_ori.shape[3]), mode='bilinear', align_corners=False)
            mask = (grd_left_imgs != 0).any(dim=1, keepdim=True).float()
            pred_depth = pred_depth * mask

            depth_paths = Data[-1]
            for j in range(len(depth_paths)):
                depth = pred_depth[j,0]
                depth = depth / depth.max() * 80
                torch.save(depth, depth_paths[j])

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=6.25e-05, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--level', type=str, default='0_2', help=' ')
    parser.add_argument('--channels', type=str, default='32_16_4', help='64_16_4 ')
    parser.add_argument('--N_iters', type=int, default=1, help='any integer')

    # parser.add_argument('--confidence', type=int, default=0, help='use confidence or not')
    parser.add_argument('--ConfGrd', type=int, default=1, help='use confidence or not for grd image')
    parser.add_argument('--ConfSat', type=int, default=0, help='use confidence or not for sat image')

    parser.add_argument('--share', type=int, default=1, help='share feature extractor for grd and sat or not '
                                                             'in translation estimation')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')
    parser.add_argument('--proj', type=str, default='geo', help='geo or CrossAttn')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    parser.add_argument('--GPS_error', type=int, default=5, help='')
    parser.add_argument('--GPS_error_coe', type=float, default=0., help='')
    parser.add_argument('--contrastive_coe', type=float, default=0., help='')

    parser.add_argument('--stage', type=int, default=1, help='0 or 1, 0 for self-supervised training, 1 for E2E training')
    parser.add_argument('--task', type=str, default='3DoF',
                        help='')

    parser.add_argument('--supervise_amount', type=float, default=1.0,
                        help='0.1, 0.2, 0.3, ..., 1')
    parser.add_argument('--name', type=str, default='test', help='')
    
    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path= restore_path = './ModelsKitti/3DoF/Stage' + str(args.stage) \
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range)  \
                + '_Nit' + str(args.N_iters) + '_' + str(args.Optimizer) + '_' + str(args.proj) \
                + '_Level' + args.level + '_Channels' + args.channels

    # if args.ConfGrd and args.stage > 0:
    #     save_path = save_path + '_ConfGrd'
    if args.ConfSat and args.stage > 0:
        save_path = save_path + '_ConfSat'

    if args.GPS_error_coe > 0 and args.stage > 0:

        save_path = save_path + '_GPSerror' + str(args.GPS_error) + '_Coe' + str(args.GPS_error_coe)


    if args.share and args.stage > 0:
        save_path = save_path + '_Share'

    if args.supervise_amount < 1 and args.stage > 0:
        save_path += '_' + str(args.supervise_amount)


    print('save_path:', save_path)
    name_path = save_path + '_' + args.name

    return save_path, restore_path, name_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    save_path, restore_path, name_path = getSavePath(args)

    net = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)

    net.to(device)

    # test1(net, args, name_path, epoch=1)
    test2(net, args, name_path, epoch=1)



def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
