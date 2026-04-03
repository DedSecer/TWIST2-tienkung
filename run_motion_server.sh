#!/bin/bash

script_dir=$(dirname $(realpath $0))
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

# Usage:
#   bash run_motion_server.sh [robot] [motion_file]
# Examples:
#   bash run_motion_server.sh unitree_g1_with_hands
#   bash run_motion_server.sh tienkung
#   bash run_motion_server.sh tienkung /abs/path/to/your_motion.pkl

robot=${1:-unitree_g1_with_hands}

# Default motion file can be overridden by arg2.

default_motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

motion_file=${2:-$default_motion_file}

# Change to deploy_real directory
cd deploy_real

# by default we use our own laptop as the redis server
redis_ip="localhost"
# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"


# Run the motion server
python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot ${robot} \
    --vis \
    --redis_ip ${redis_ip}
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
