#!/bin/bash

tmux new -s gaussian1 'python train_gaussian_map.py; tmux kill-session -t mysession'


export LD_LIBRARY_PATH=/home/wangqw/.conda/envs/cvlnet2/lib:$LD_LIBRARY_PATH
# 运行 Python 脚本
python "./train_KITTI_weak.py" \
    --rotation_range 10 \
    --stage 1 \
    --share 1 \
    --level 1 \
    --ConfGrd 0 \
    --contrastive_coe 1 \
    --name 'test'

python "./train_KITTI_weak.py" \
    --rotation_range 10 \
    --stage 0 \
    --share 1 \
    --level 2 \
    --name "128*512*16" \
    --epochs 8 \
    --batch_size 12

python "./train_KITTI_weak.py" \
  --rotation_range 10 \
  --stage 2 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "feat32_no_rgb" \
  --batch_size 12 \
  --epochs 8
  
  #6.25e-5 cos GPS
  python "./train_KITTI_weak_nips.py" \
    --rotation_range 0 \
    --stage 4 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_offset_0.5_confidence_original_gpv_1" \
    --batch_size 8 \
    --epochs 10 \
    --test 0 \
    --visualize 0
    
  python "./train_KITTI_weak_seq.py" \
    --rotation_range 0 \
    --stage 4 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_offset_0.5_seq3_6.5e-5" \
    --batch_size 8 \
    --epochs 10 \
    --test 0 \
    --visualize 0 \
    --sequence 3


  #6.25e-5 cos
  python "./train_KITTI_weak.py" \
    --rotation_range 40 \
    --stage 3 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_GPS_ori_range40" \
    --batch_size 12 \
    --epochs 5

  python "./train_KITTI_weak.py" \
    --rotation_range 40 \
    --stage 3 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_no_GPS_ori_range40" \
    --batch_size 12 \
    --epochs 5

python "./train_KITTI_weak.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "op_as_confidence" \
  --batch_size 8 \
  --epochs 10

python "./train_KITTI_weak.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "original_confidence" \
  --batch_size 8 \
  --epochs 10

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --area "same" \
  --name '20face_160to160'

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --area "same" \
  --name '20face_80to160'

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --area "same" \
  --name '6face_160to160'

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'resnet_vigor' \
  --batch_size 8 \
  --epochs 15 \
  --grd_res 80 \
  --test 0 \
  --share 0 \
  --lr 6.5e-5

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_1.0' \
  --batch_size 12 \
  --epochs 15 \
  --grd_res 80 \
  --test 0 \
  --share 0 \
  --lr 6.5e-5

python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Gaussian" \
  --area "same" \
  --name 'omni_scene' \
  --batch_size 32

python "./train_KITTI_weak_direct.py" \
  --rotation_range 0 \
  --stage 3 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "forward_mapping_GPS" \
  --batch_size 16 \
  --epochs 5