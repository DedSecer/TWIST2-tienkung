#!/bin/bash

# Usage: bash train_stage1_amp.sh <experiment_id> <device>

# Activate twist2 conda environment
source /home/vega/anaconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}

cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_priv_mimic_amp"
proj_name="${robot_name}_priv_mimic_amp"

python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}"
