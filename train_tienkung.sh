#!/bin/bash

# Usage: bash train_tienkung.sh <experiment_id> <device>

# Activate twist2 conda environment
source /home/vega/anaconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}
# Example: bash train_tienkung.sh 0330_tienkung_twist2 cuda:0

cd legged_gym/legged_gym/scripts

robot_name="tienkung"
exptid=$1
device=$2
teacher_exptid=$3

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "${teacher_exptid}" \
                # --resume \
                # --debug \
