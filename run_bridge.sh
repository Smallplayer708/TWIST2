#!/bin/bash

# TWIST2 -> OrcaLab bridge launcher (低层 RL 策略在 OrcaLab 仿真侧推理).
# 需 `orcalab_lerobot` conda 环境。
#
# 用法:
#   bash run_bridge.sh                     # 全身模式 + 腿部 A/C 调参
#   bash run_bridge.sh --fix_feet          # 双臂遥操作（焊接下肢，锁腿）
#   bash run_bridge.sh --verbose           # 开诊断日志
#
# 无头自检（5s 自动退出，另起终端）:
#   cd /home/user/OrcaManipulation/src/examples/dataCollection
#   python bridge_twist2_to_orcalab.py --policy $(pwd)/../../../../TWIST2/assets/ckpts/twist2_1017_20k.onnx \
#       --local_xml "$(ls -t ~/.orcagym/tmp/*.xml | head -1)" --no_render --fix_feet --device cpu --selftest

source ~/miniconda3/bin/activate orcalab_lerobot

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# 方法 A：腿部 12 关节 PD 增益缩放。OrcaLab 笔记记着单独 2.0 会闭环抖动，起步 1.5。
LEG_PD_GAIN=${LEG_PD_GAIN:-1.5}
# 方法 C：腿部 PD 目标 EMA 低通（0=关，0.5~0.7 建议）。
LEG_EMA_ALPHA=${LEG_EMA_ALPHA:-0.6}

BRIDGE_DIR=/home/user/OrcaManipulation/src/examples/dataCollection

if [ ! -f "${ckpt_path}" ]; then
    echo "Error: policy not found: ${ckpt_path}"
    exit 1
fi
if [ ! -f "${BRIDGE_DIR}/bridge_twist2_to_orcalab.py" ]; then
    echo "Error: bridge not found: ${BRIDGE_DIR}/bridge_twist2_to_orcalab.py"
    exit 1
fi

cd ${BRIDGE_DIR}

python bridge_twist2_to_orcalab.py \
    --policy ${ckpt_path} \
    --device cpu \
    --leg_pd_gain ${LEG_PD_GAIN} \
    --leg_ema_alpha ${LEG_EMA_ALPHA} \
    "$@"
