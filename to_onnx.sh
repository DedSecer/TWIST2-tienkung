#!/bin/bash

# Script to convert student policy with future motion support to ONNX

# Activate twist2 conda environment
source /home/vega/anaconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}

# bash to_onnx.sh $YOUR_POLICY_PATH

ckpt_path=$1

cd legged_gym/legged_gym/scripts

# Run the correct ONNX conversion script
python save_onnx.py --ckpt_path ${ckpt_path}