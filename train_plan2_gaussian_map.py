import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import argparse
import torch.optim as optim
import os
import scipy.io as scio

from dataLoader.KITTI_dataset_gaussian import load_train_data, load_test1_data
from kitti_image_model_plan2_gaussian import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gaussian_map')
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warm_up_steps', type=float, default=2000)    
    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--rotation_range', type=float, default=0., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')
    parser.add_argument('--predict_height', type=int, default=1., help='whether to predict height')
    parser.add_argument('--feature_forward_project', type=int, default=0, help='test with trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--root', type=str, default='/data/dataset/KITTI/', help='test with trained model')
    return parser.parse_args()

def getSavePath(args):
    save_path = './ModelsKitti/2DoF/' + args.name

    # if args.use_uncertainty:
    #     save_path = save_path + '_Uncertainty'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path:', save_path)

    return save_path

def train(model, lr, args, save_path):
    for epoch in range(args.epochs):
        
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / args.warm_up_steps,
            end_factor=1.0,
            total_iters=args.warm_up_steps,
        )

        optimizer.zero_grad()

        train_loader = load_train_data(mini_batch, root=args.root)
        print('batch_size:', mini_batch, '\n num of batches:', len(train_loader))

        for Loop, Data in enumerate(train_loader, 0):
            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_height, project_map, sat_height, grd_depth = [item.to(device) for item in Data[:-1]]

            optimizer.zero_grad()
            # corr_loss, mse_loss = model.feature_map(sat_map, grd_left_imgs, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            # loss = corr_loss + mse_loss
            model.gaussian_init(sat_map, grd_left_imgs, project_map, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            loss = model.gaussian_map()
            loss.backward()
            # 打印每个参数的梯度
            # for name, param in model.named_parameters():
            #     print(f"Parameter: {name}, Gradient: {param.shape}")
            optimizer.step()
            optimizer.zero_grad()

            # 更新学习率
            scheduler.step()
            model.global_step = model.global_step + sat_map.shape[0]

            if Loop % 10 == 9:  #
                model.gaussian_map(deterministic=True)
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Loss: ' + str(loss.item()))

        print('Save Model ...')
        torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))
        # test1(model, args, save_path, epoch)
    print('Finished Training')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    np.random.seed(2022)
    args = parse_args()
    mini_batch = args.batch_size

    save_path = getSavePath(args)
    lr = args.lr

    model = Model(args).to(device)

    train(model, lr, args, save_path)