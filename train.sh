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
  --stage 2 \
  --share 1 \
  --level 1 \
  --ConfGrd 0 \
  --contrastive_coe 1 \
  --name '64to256' \
  --batch_size 24 \
  --epochs 10

  python "./train_KITTI_weak.py" \
    --rotation_range 10 \
    --stage 3 \
    --share 1 \
    --level 1 \
    --ConfGrd 0 \
    --contrastive_coe 1 \
    --name 'fix_mask'

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
