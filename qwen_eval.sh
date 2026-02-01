#!/bin/bash
source ~/ShenxuC/miniconda3/etc/profile.d/conda.sh
conda activate python310
cd ~/ShenxuC/diffuTime


echo "Starting hallucination evaluation using Qwen8B..."

CUDA_VISIBLE_DEVICES=3 python ./DLM_generate/metric_qwen.py > logs/qwen_eval.log 2>&1

echo "All tasks completed."