#!/usr/bin/env python3
"""Replay a recorded teleop session through the sim2sim policy loop and
measure root forward progress vs the reference, in the CURRENT mujoco env.

Usage:
  python compare_tools/replay_policy.py --log logs/run1.json --xml ../assets/g1/g1_sim2sim_29dof.xml \
      --policy ../assets/ckpts/twist2_1017_20k.onnx --device cpu
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server_low_level_g1_sim import load_onnx_policy

STIFFNESS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
])
DAMPING = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    4, 4, 4,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
    5, 5, 5, 5, 0.2, 0.2, 0.2,
])
TORQUE_LIMITS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    150, 150, 150,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
    40, 40, 40, 40, 4.0, 4.0, 4.0,
])
ACTION_SCALE = 0.5
DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
])
ANKLE_IDX = [4, 5, 10, 11]

N_MIMIC = 35
N_PROPRIO = 92
N_OBS_SINGLE = N_MIMIC + N_PROPRIO
HISTORY_LEN = 10
TOTAL_OBS = N_OBS_SINGLE * (HISTORY_LEN + 1) + N_MIMIC


def quat_to_euler(quat):
    qw, qx, qy, qz = quat
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max_sec", type=float, default=40.0)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--synthetic_vx", type=float, default=0.0,
                    help="If >0, replace mimic xy-vel/yaw with a synthetic constant "
                         "forward speed reference (closed-loop gain test)")
    ap.add_argument("--z_override", type=float, default=-1.0,
                    help="Override mimic root z (default -1 = use teleop value, "
                         "e.g. 0.79 matches the sim pelvis height)")
    ap.add_argument("--legs_fixed_ref", action="store_true",
                    help="Use standing-pose leg reference instead of teleop gait")
    args = ap.parse_args()

    log = json.load(open(args.log))
    frames = log["frames"]
    a = args.start_frame
    b = len(frames) if args.end_frame < 0 else args.end_frame
    frames = frames[a:b]
    n = min(len(frames), int(args.max_sec * args.fps))
    frames = frames[:n]
    print(f"[replay] {n} frames (from {a} to {a + n})")

    policy = load_onnx_policy(args.policy, args.device)
    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    decimation = int(round(1.0 / (args.fps * model.opt.timestep)))
    print(f"[replay] decimation={decimation} (policy {args.fps}Hz)")

    history = [np.zeros(N_OBS_SINGLE, dtype=np.float32) for _ in range(HISTORY_LEN)]
    last_action = np.zeros(29, dtype=np.float32)
    pd_target = DEFAULT_DOF_POS.copy()
    import torch

    sim_root_xy = [data.qpos[0:2].copy()]
    ref_root_xy = np.array([f["teleop_state"]["root_xyz"][:2] for f in frames[:n]])

    for i in range(n):
        fr = frames[i]
        tel = fr["teleop_state"]
        # rebuild mimic obs from teleop state + a synthetic root vel (0 unless in raw)
        # We do NOT have the exact mimic_obs in the log; rebuild it from teleop qpos deltas
        # stored states give us dof + root; use the recorded sim2sim state for proprio.
        dof_target = np.array(tel["dof_pos"], dtype=float)
        if args.legs_fixed_ref:
            dof_target[:15] = DEFAULT_DOF_POS[:15]
        if args.synthetic_vx > 0:
            # closed-loop forward-speed gain test: constant vx reference,
            # legs keep the reference gait but the speed command is synthetic
            v_xy = np.array([args.synthetic_vx, 0.0])
            yaw_vel = 0.0
        elif i == 0:
            v_xy = np.zeros(2)
            yaw_vel = 0.0
        else:
            prev_frame = frames[i - 1]
            prev = prev_frame["teleop_state"]
            dt = max(fr["t_tel_ms"] - prev_frame["t_tel_ms"], 1) / 1000.0
            v_xy = (np.array(tel["root_xyz"][:2]) - np.array(prev["root_xyz"][:2])) / dt
            yaw_prev = quat_to_euler(prev["root_quat"])[2]
            yaw_cur = quat_to_euler(tel["root_quat"])[2]
            yaw_vel = (yaw_cur - yaw_prev) / dt
        z_ref = tel["root_xyz"][2:3] if args.z_override <= 0 else [args.z_override]
        mimic = np.concatenate([
            v_xy, z_ref,
            quat_to_euler(tel["root_quat"])[:2], [yaw_vel],
            dof_target,
        ]).astype(np.float32)

        dof_pos = data.qpos[7:36].copy()
        dof_vel = data.qvel[6:35].copy()
        quat = data.qpos[3:7].copy()
        ang_vel = data.qvel[3:6].copy()
        rpy = quat_to_euler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[ANKLE_IDX] = 0.0
        proprio = np.concatenate([
            ang_vel * 0.25, rpy[:2],
            (dof_pos - DEFAULT_DOF_POS),
            obs_dof_vel * 0.05,
            last_action,
        ]).astype(np.float32)
        obs_full = np.concatenate([mimic, proprio]).astype(np.float32)
        obs_hist = np.array(history).flatten()
        history.append(obs_full)
        history.pop(0)
        obs_buf = np.concatenate([obs_full, obs_hist, mimic])
        assert obs_buf.shape[0] == TOTAL_OBS, obs_buf.shape

        with torch.no_grad():
            raw_action = policy(torch.from_numpy(obs_buf).float().unsqueeze(0)).numpy().squeeze()
        last_action = raw_action
        raw_action = np.clip(raw_action, -10.0, 10.0)
        pd_target = raw_action * ACTION_SCALE + DEFAULT_DOF_POS

        for _ in range(decimation):
            dof_pos = data.qpos[7:36]
            dof_vel = data.qvel[6:35]
            torque = (pd_target - dof_pos) * STIFFNESS - dof_vel * DAMPING
            torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        sim_root_xy.append(data.qpos[0:2].copy())

    sim_root_xy = np.array(sim_root_xy)
    sim_disp = np.linalg.norm(sim_root_xy - sim_root_xy[0], axis=1)
    ref_disp = np.linalg.norm(ref_root_xy - ref_root_xy[0], axis=1)
    print(f"[replay] ref 前进: {ref_disp[-1]:.2f} m  sim 前进: {sim_disp[-1]:.2f} m  比值: {sim_disp[-1]/max(ref_disp[-1],1e-6):.2f}")
    # 分段时间
    seg = 500
    for a in range(0, n - seg, seg):
        b = a + seg
        print(f"  t={a/args.fps:4.1f}s: ref {ref_disp[b]:.2f} m, sim {sim_disp[a]+0:.2f}->{sim_disp[b]:.2f} m")
    print(f"[replay] sim root z min/max: {data.qpos[2]:.3f}")


if __name__ == "__main__":
    main()
