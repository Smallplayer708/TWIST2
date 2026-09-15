

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=eno1

# Leg jitter tuning (ported from the sim path):
#   leg_ema_alpha : EMA low-pass on the 12 leg PD targets (0=off, 0.5~0.7 recommended)
#   leg_pd_gain   : scale the 12 leg joints kp/kd (1.0=off, try 1.5~2.0)
leg_ema_alpha=0.6
leg_pd_gain=1.5

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --leg_ema_alpha ${leg_ema_alpha} \
    --leg_pd_gain ${leg_pd_gain}
    # optional: --smooth_body 0.5   --record_proprio
