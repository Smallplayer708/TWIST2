#!/bin/bash
source ~/miniconda3/bin/activate twist2
cd /home/user/TWIST2/deploy_real
python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof_with_hands.xml \
    --policy ../assets/ckpts/twist2_1017_20k.onnx \
    --device cpu --pipeline_mode 1 --no_viewer 1 \
    --policy_frequency 100 --limit_fps 0
