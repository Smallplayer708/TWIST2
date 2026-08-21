#!/bin/bash
source ~/miniconda3/bin/activate gmr
cd /home/user/TWIST2/deploy_real
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
    --actual_human_height 1.6 --redis_ip localhost \
    --target_fps 100 --measure_fps 1 --fixed_lower_body
