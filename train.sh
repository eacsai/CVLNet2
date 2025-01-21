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
  --name "feat32_128*512" \
  --batch_size 4 \
  --epochs 8
  
  #6.25e-5 cos GPS
  python "./train_KITTI_weak.py" \
    --rotation_range 10 \
    --stage 3 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_GPS" \
    --batch_size 12 \
    --epochs 10

  #6.25e-5 cos
  python "./train_KITTI_weak.py" \
    --rotation_range 10 \
    --stage 3 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_no_GPS" \
    --batch_size 12 \
    --epochs 10

python "./train_KITTI_weak.py" \
  --rotation_range 10 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "feat32_dpt_best" \
  --batch_size 12 \
  --epochs 4

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
  --rotation_range 180 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name '20face_40_same_180' \
  --batch_size 16 \
  --epochs 15 \
  --grd_res 40