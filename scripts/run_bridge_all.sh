#!/bin/bash
LOG="/home/user/TWIST2/deploy_real/twist2_orca_bridge.log"

echo "============================================"
echo "  TWIST2 + OrcaLab 双臂遥操作 启动脚本"
echo "============================================"
echo ""
echo "  前提: OrcaLab GUI 已加载场景 + 点击运行 (gRPC 50051)"
echo "        PICO 已 USB 连接 + adb reverse tcp:8001 tcp:8001"
echo "        redis-server 已运行"
echo ""

if command -v tmux &>/dev/null; then
    SESSION="twist2_orca"
    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"
    tmux rename-window -t "$SESSION:0" "Bridge"
    tmux split-window -h -t "$SESSION:0"
    tmux split-window -v -t "$SESSION:0.1"

    # 面板0.2 (右下): TWIST2 遥操作 (先启动, 等 PICO)
    tmux send-keys -t "$SESSION:0.2" 'echo "=== TWIST2 Teleop ===" && bash /home/user/TWIST2/scripts/run_twist2_teleop.sh' Enter
    sleep 3
    # 面板0.1 (右上): TWIST2 管道
    tmux send-keys -t "$SESSION:0.1" 'echo "=== TWIST2 Pipeline ===" && bash /home/user/TWIST2/scripts/run_twist2_pipeline.sh' Enter
    sleep 2
    # 面板0.0 (左上): OrcaLab 遥操作 (最后, 确保 Redis 有数据)
    tmux send-keys -t "$SESSION:0.0" 'echo "=== OrcaLab Teleop ===" && bash /home/user/TWIST2/scripts/run_orcalab_teleop.sh' Enter

    tmux new-window -t "$SESSION" -n "Logs"
    tmux send-keys -t "$SESSION:1" "tail -f $LOG" Enter

    tmux select-window -t "$SESSION:0"
    tmux attach-session -t "$SESSION"
else
    echo ""
    echo "tmux 未安装。请手动在 3 个终端中按顺序运行："
    echo ""
    echo "  ── 终端1: TWIST2 Teleop ──"
    echo "  bash /home/user/TWIST2/scripts/run_twist2_teleop.sh"
    echo ""
    echo "  ── 终端2: TWIST2 Pipeline ──"
    echo "  bash /home/user/TWIST2/scripts/run_twist2_pipeline.sh"
    echo ""
    echo "  ── 终端3: OrcaLab Teleop (最后启动) ──"
    echo "  bash /home/user/TWIST2/scripts/run_orcalab_teleop.sh"
    echo ""
    echo "  ── 日志监控 ──"
    echo "  tail -f $LOG"
    echo ""
    echo "安装 tmux: sudo apt install tmux"
fi
