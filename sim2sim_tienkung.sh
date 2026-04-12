#!/bin/bash
# Tienkung sim2sim verification script
# Usage: bash sim2sim_tienkung.sh [onnx_path]
#
# If no onnx_path argument is given, defaults to the most recently exported model.
# Example:
#   bash sim2sim_tienkung.sh /path/to/your_tienkung_policy.onnx

SCRIPT_DIR=$(dirname $(realpath $0))

# Default: use provided argument or a placeholder
ckpt_path=${1:-${SCRIPT_DIR}/assets/ckpts/twist2_0406_pdgain_tienkung_30k.onnx}

cd deploy_real

python server_low_level_tienkung_sim.py \
    --xml ../assets/tienkung/mjcf/tienkung.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 50 \
    --limit_fps 1 \
    # --record_proprio \
