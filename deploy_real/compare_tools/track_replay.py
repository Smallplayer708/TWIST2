#!/usr/bin/env python3
"""Offline open-loop tracking test: replay teleop dof_pos targets through the
MuJoCo PD controller and measure per-joint tracking error for a given XML.
Quantifies the solver/contact dynamics gap between MuJoCo and Isaac Gym.

Usage:
  python compare_tools/track_replay.py --log logs/run1.json --xml A.xml --xml B.xml
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

DEFAULT_DOF_POS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
])

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
    "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def track(xml_path, target_dof_seq, target_root_seq, dt_sim=0.001,
          decimation=20, reset_every=None, action_scale=0.5):
    """Replay targets with the deploy PD controller; return per-frame errors."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt_sim
    data = mujoco.MjData(model)
    n = len(target_dof_seq)
    errs = []
    dof_idx = 7
    last_pd_target = DEFAULT_DOF_POS.copy()
    for i in range(n):
        # deploy-equivalent: target arrives every decimation steps
        if i % decimation == 0:
            raw_action = (target_dof_seq[i] - DEFAULT_DOF_POS) / action_scale
            raw_action = np.clip(raw_action, -10.0, 10.0)
            last_pd_target = raw_action * action_scale + DEFAULT_DOF_POS
        dof_pos = data.qpos[dof_idx:dof_idx + 29]
        dof_vel = data.qvel[6:6 + 29]
        torque = (last_pd_target - dof_pos) * STIFFNESS - dof_vel * DAMPING
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        errs.append(np.abs(last_pd_target - dof_pos))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--xml", action="append", required=True)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--decimation", type=int, default=20)
    args = ap.parse_args()

    log = json.load(open(args.log))
    frames = log["frames"]
    # use teleop dof as the PD target stream (drop first 50, take 600)
    target_dof = np.array([f["teleop_state"]["dof_pos"] for f in frames[50:650]])
    print(f"targets: {target_dof.shape}")

    results = {}
    for xml in args.xml:
        name = os.path.basename(xml)
        errs = track(xml, target_dof, None, dt_sim=args.dt,
                     decimation=args.decimation)
        # skip warmup
        errs = errs[100:]
        rmse = np.degrees(np.sqrt((errs ** 2).mean(axis=0)))
        results[name] = rmse
        leg = rmse[:12].mean()
        arm = rmse[15:].mean()
        print(f"\n=== {name} ===")
        print(f"  leg mean RMSE: {leg:.2f} deg   arm mean RMSE: {arm:.2f} deg")
        order = np.argsort(-rmse)[:6]
        for idx in order:
            print(f"  {JOINT_NAMES[idx]:24s} {rmse[idx]:6.2f} deg")

    if len(results) == 2:
        a, b = list(results.keys())
        print(f"\n=== {a} vs {b} (leg RMSE ratio: "
              f"{results[a][:12].mean()/results[b][:12].mean():.3f}) ===")


if __name__ == "__main__":
    main()
