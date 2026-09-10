# NOTE: run with the `gmr` conda env (python 3.10, mujoco 3.11.0).
# The `twist2` env is python 3.8 and cannot install mujoco > 3.2.3;
# it is kept only for isaacgym training.
#   conda activate gmr && bash sim2sim.sh

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cpu \
    --measure_fps 1 \
    --policy_frequency 50 \
    --limit_fps 1 \
    --render_interval 2 \
    --leg_pd_gain 2.0 \
    --leg_ema_alpha 0.6 \
    # --record_proprio \
