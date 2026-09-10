#!/bin/bash
# TWIST2 sim2sim — runs ONNX policy in MuJoCo simulation
#
# Modes (via --mode):
#   free      upstream behaviour: policy 100Hz, no leg tuning, no lower-body lock
#   fix_feet  arm-only: weld pelvis + zero leg/torso + stiffer arms (x2.5 PD)
#   tuned     leg-tracking improvements: 50Hz + render_interval + leg_pd_gain + leg_ema
#
# Usage:
#   bash sim2sim.sh                # free mode (default)
#   bash sim2sim.sh --mode fix_feet
#   bash sim2sim.sh --mode tuned
#   bash sim2sim.sh --redis_verify # check Redis data, no sim (passed through)
#
# NOTE: run with the `gmr` conda env (python 3.10, mujoco 3.11.0).
# The `twist2` env is python 3.8 and cannot install mujoco > 3.2.3.
#   conda activate gmr && bash sim2sim.sh [--mode tuned]

set -euo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

source ~/miniconda3/bin/activate gmr

MODE="free"
DEVICE="cpu"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"; shift 2 ;;
        --mode=*)
            MODE="${1#*=}"; shift ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --device=*)
            DEVICE="${1#*=}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "${MODE}" in
    free)
        POLICY_FREQ=100
        MODE_ARGS=()
        ;;
    fix_feet)
        POLICY_FREQ=100
        MODE_ARGS=(--fix_feet --arm_pd_gain 2.5)
        ;;
    tuned)
        POLICY_FREQ=50
        MODE_ARGS=(--render_interval 2 --leg_pd_gain 2.0 --leg_ema_alpha 0.6)
        ;;
    *)
        echo "Error: unknown --mode '${MODE}' (use free|fix_feet|tuned)" >&2
        exit 1
        ;;
esac

cd "${SCRIPT_DIR}/deploy_real"

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy "${ckpt_path}" \
    --device "${DEVICE}" \
    --measure_fps 1 \
    --policy_frequency "${POLICY_FREQ}" \
    --limit_fps 1 \
    "${MODE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
