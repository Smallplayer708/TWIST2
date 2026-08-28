#!/bin/bash
# TWIST2 sim2sim — runs ONNX policy in MuJoCo simulation
# Free mode (default, no args) is byte-equivalent to upstream TWIST2:
# same xml/policy/policy_frequency/limit_fps, no lower-body restriction.
# Usage:
#   bash sim2sim.sh                  # free mode (default): upstream behaviour
#   bash sim2sim.sh --fix_feet       # fixed lower body: weld pelvis + zero leg/torso actions
#   bash sim2sim.sh --redis_verify   # check Redis data, no sim
# NOTE: onnxruntime here only has CPUExecutionProvider (no CUDA), so --device
# makes no difference — the policy always runs on CPU (see load_onnx_policy).

set -euo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

source ~/miniconda3/bin/activate gmr

DEVICE="cpu"
FIX_FEET=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"; shift 2 ;;
        --device=*)
            DEVICE="${1#*=}"; shift ;;
        --fix_feet)
            FIX_FEET="--fix_feet"; shift ;;
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
    ${FIX_FEET} \
    "${EXTRA_ARGS[@]}"
