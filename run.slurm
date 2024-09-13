#!/bin/bash
#SBATCH -J a40_cvlnet_inverse
#SBATCH -p ShangHAI
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=all
#SBATCH --mail-user=v-wangqw@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH --output=a40_cvlnet_inverse.out
#SBATCH --error=a40_cvlnet_inverse.err
#SBATCH -A shiyj2-ShangHAI

python -u ./train_plan2_inverse_map.py