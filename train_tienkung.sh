#!/bin/bash

# Usage: bash train_tienkung.sh <experiment_id> <device>
# Example: bash train_tienkung.sh 0330_tienkung_twist2 cuda:0

cd legged_gym/legged_gym/scripts

robot_name="tienkung"
exptid=$1
device=$2

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                # --resume \
                # --debug \
