#!/bin/bash
# TWIST2 real-time teleop (PICO -> GMR retarget -> Redis -> policy)
#
# Modes (via --mode):
#   free      upstream behaviour: full body, smooth damping (5e-1), no leg tuning
#   fix_feet  arm-only: lock lower body to standing pose (--fixed_lower_body)
#   tuned     full body with leg-tracking improvements: damping 1e-1 + yaw gain + leg smoothing
#
# Usage:
#   bash teleop.sh                # free mode (default)
#   bash teleop.sh --mode fix_feet
#   bash teleop.sh --mode tuned

# sudo ufw disable

source ~/miniconda3/bin/activate gmr

cd deploy_real

# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"
# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.6

MODE="free"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"; shift 2 ;;
        --mode=*)
            MODE="${1#*=}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "${MODE}" in
    free)
        MODE_ARGS=()
        ;;
    fix_feet)
        MODE_ARGS=(--fixed_lower_body)
        ;;
    tuned)
        MODE_ARGS=(--retarget_damping 1.0e-1 --yaw_gain 1.5 --leg_smooth_alpha 0.8)
        ;;
    *)
        echo "Error: unknown --mode '${MODE}' (use free|fix_feet|tuned)" >&2
        exit 1
        ;;
esac

python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --hand_step 0.02 \
             "${MODE_ARGS[@]}" \
             "${EXTRA_ARGS[@]}"
