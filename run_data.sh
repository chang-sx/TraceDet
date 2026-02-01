#!/bin/bash
source ~/ShenxuC/miniconda3/etc/profile.d/conda.sh
conda activate python310
cd ~/ShenxuC/diffuTime





CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task test --generate_task entropy --gen_length 64  > logs/triviaqa_test_Instruct_64_ent.log 2>&1 &




wait