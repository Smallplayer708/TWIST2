#!/bin/bash
source ~/miniconda3/bin/activate orcalab_lerobot
cd /home/user/OrcaManipulation/src/examples/dataCollection
python g1_pick_teleop_twist2.py \
    --task_config twist2_teleop.yaml \
    --scene_json g1_pick_button.json \
    --lerobot_out /home/user/g1_pick_data \
    --repo_id local/g1_pick_teleop \
    --fps 20 --clock wall \
    --orcagym_addr localhost:50051 \
    --log_file /home/user/OrcaManipulation/src/examples/dataCollection/logs/orcalab_terminal.log
