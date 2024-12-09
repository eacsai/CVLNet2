import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import argparse
import torch.optim as optim
import os
import scipy.io as scio

from dataLoader.KITTI_dataset_forward import load_train_data, load_test1_data
from kitti_image_model_plan2 import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='feature_forward_map_original')
    parser.add_argument('--epochs', type=int, default=7) 
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
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


def test1(model, args, save_path, epoch):
    model.eval()
    dataloader = load_test1_data(mini_batch, args.root, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

    print('batch_size:', mini_batch, '\n num of batches:', len(dataloader))
    pred_lons = []
    pred_lats = []

    gt_lons = []
    gt_lats = []

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_height, project_map, sat_height, grd_depth = [item.to(device) for item in Data[:-1]]
            pred_u, pred_v = model.feature_map(sat_map, grd_left_imgs, project_map, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='test')

            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)

            if i % 20 == 0:
                print(i)

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)

    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    scio.savemat(os.path.join(save_path, 'test1_result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats,
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
    
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)

    metrics = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Test1 results:')
    
    print('Distance average: (init, pred)', np.mean(init_dis), np.mean(distance))
    print('Distance median: (init, pred)', np.median(init_dis), np.median(distance))

    print('Lateral average: (init, pred)', np.mean(np.abs(gt_lats)), np.mean(diff_lats))
    print('Lateral median: (init, pred)', np.median(np.abs(gt_lats)), np.median(diff_lats))

    print('Longitudinal average: (init, pred)', np.mean(np.abs(gt_lons)), np.mean(diff_lons))
    print('Longitudinal median: (init, pred)', np.median(np.abs(gt_lons)), np.median(diff_lons))

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()

    model.train()
    return

def train(model, lr, args, save_path):
    for epoch in range(args.epochs):
        
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        optimizer.zero_grad()

        train_loader = load_train_data(mini_batch, root=args.root)
        print('batch_size:', mini_batch, '\n num of batches:', len(train_loader))

        for Loop, Data in enumerate(train_loader, 0):
            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_height, project_map, sat_height, grd_depth = [item.to(device) for item in Data[:-1]]

            optimizer.zero_grad()

            # corr_loss, mse_loss = model.feature_map(sat_map, grd_left_imgs, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            # loss = corr_loss + mse_loss
            loss = model.feature_map(sat_map, grd_left_imgs, project_map, grd_depth, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if Loop % 10 == 9:  #
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Loss: ' + str(loss.item()))

        print('Save Model ...')
        torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))
        test1(model, args, save_path, epoch)
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
    if args.test:
        # model.load_state_dict(torch.load(os.path.join(save_path, 'model_4.pth')))
        test1(model, args, save_path, epoch=4)
    else:
        train(model, lr, args, save_path)