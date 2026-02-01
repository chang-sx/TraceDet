#!/bin/bash
source ~/ShenxuC/miniconda3/etc/profile.d/conda.sh
conda activate python310
cd ~/ShenxuC/diffuTime

CUDA_VISIBLE_DEVICES=4  python train_tuning.py --num_repeats 3 --device cuda --dataset commonsenseqa   > logs/commonsenseqa_l2_tuning.log 2>&1 &
CUDA_VISIBLE_DEVICES=5  python train_tuning.py --num_repeats 3 --device cuda --dataset triviaqa  > logs/triviaqa_l2_tuning.log 2>&1 &
CUDA_VISIBLE_DEVICES=6  python train_tuning.py --num_repeats 3 --device cuda --dataset hotpotqa   > logs/hotpotqa_l2_tuning.log 2>&1 &


wait