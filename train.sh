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

python "./train_KITTI_weak_nips_orienternet.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "orienternet_weakly_GPS" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0 

python "./train_KITTI_weak_nips_vfa.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "vfa_weakly_GPS" \
  --batch_size 8 \
  --epochs 10 \
  --test 0 \
  --visualize 0 


python "./train_KITTI_weak_weather.py" \
  --rotation_range 0 \
  --stage 4 \
  --share 1 \
  --level 1 \
  --ConfGrd 1 \
  --contrastive_coe 1 \
  --name "feat32_offset_0.5_confidence_original" \
  --batch_size 8 \
  --epochs 10 \
  --test 1 \
  --visualize 0 



  #6.25e-5 cos GPS
  python "./train_KITTI_weak_nips.py" \
    --rotation_range 0 \
    --stage 4 \
    --share 1 \
    --level 1 \
    --ConfGrd 1 \
    --contrastive_coe 1 \
    --name "feat32_FineGPS" \
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
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_0.3_3.0_70_1.25e-4_depth' \
  --batch_size 32 \
  --test 1 \
  --lr 1.25e-4 \
  --amount 0.01


python "./train_vigor_2DoF.py" \
  --rotation_range 0 \
  --share 0 \
  --ConfGrd 1 \
  --level 1 \
  --Supervision "Weakly" \
  --area "same" \
  --name 'vigor_0.3_3.0_80_1.25e-4_depth' \
  --batch_size 8 \
  --test 0 \
  --lr 1.25e-4 \
  --epoch 15

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