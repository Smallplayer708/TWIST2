#!/bin/bash
# TWIST2 sim2sim — runs ONNX policy in MuJoCo simulation
# Usage: bash sim2sim.sh                (default: cpu)
#        bash sim2sim.sh --device cuda   (GPU mode)
#        bash sim2sim.sh --redis_verify (check Redis data, no sim)

set -euo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

source ~/miniconda3/bin/activate gmr

DEVICE="cpu"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"; shift 2 ;;
        --device=*)
            DEVICE="${1#*=}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

cd "${SCRIPT_DIR}/deploy_real"

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy "${ckpt_path}" \
    --device "${DEVICE}" \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    --fix_feet \
    "${EXTRA_ARGS[@]}"
