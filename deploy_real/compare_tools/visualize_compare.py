#!/usr/bin/env python3
"""
Teleop-vs-sim2sim dual-robot visualizer.

Shows both robots in one MuJoCo scene (teleop = blue, sim2sim = red, both
semi-transparent) with initial root frames aligned, so the motion difference
(lower body / root drift) is visible directly.

Two modes:
  --mode live     read compare_teleop_state / compare_sim2sim_state from Redis
  --mode replay   read a .json log produced by compare_record.py

Usage (conda env: twist2):
  python compare_tools/visualize_compare.py --mode live
  python compare_tools/visualize_compare.py --mode replay --log logs/run1.json
"""

import argparse
import json
import os
import sys
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import redis
from scipy.spatial.transform import Rotation as R

# make the package importable regardless of the caller's working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_tools.compare_record import (
    JOINT_NAMES, REDIS_TELEOP_KEY, REDIS_SIM2SIM_KEY,
    make_align_transform, quat_to_xyzw, xyzw_to_quat, parse_state,
)

ASSET_DIR = "/home/user/TWIST2/assets/g1"
DUAL_XML = ASSET_DIR + "/compare_dual.xml"

NQ_PER_ROBOT = 36
TELEOP_SLICE = slice(0, NQ_PER_ROBOT)
SIM2SIM_SLICE = slice(NQ_PER_ROBOT, 2 * NQ_PER_ROBOT)


def color_robots(model):
    """teleop -> blue translucent, sim2sim -> red translucent."""
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        rgba = model.geom_rgba[i].copy()
        if body_name.startswith("teleop_"):
            rgba[:3] = (0.25, 0.45, 0.95)
            rgba[3] = min(rgba[3], 0.55)
        elif body_name.startswith("sim2sim_"):
            rgba[:3] = (0.95, 0.25, 0.20)
            rgba[3] = min(rgba[3], 0.55)
        model.geom_rgba[i] = rgba


def state_to_qpos(st):
    return np.concatenate([
        np.asarray(st["root_xyz"], dtype=float),
        np.asarray(st["root_quat"], dtype=float),
        np.asarray(st["dof_pos"], dtype=float),
    ])


def aligned_state_to_qpos(st, align):
    """Map a sim2sim state into the teleop frame (initial-frame aligned)."""
    p, q = align(st["root_xyz"], st["root_quat"])
    return np.concatenate([
        p, q, np.asarray(st["dof_pos"], dtype=float),
    ])


def live_mode(viewer, model, data, client):
    print("[viz] live mode: waiting for teleop + sim2sim states...")
    align = None
    last_print = 0.0
    while viewer.is_running():
        tel = parse_state(client.get(REDIS_TELEOP_KEY))
        sim = parse_state(client.get(REDIS_SIM2SIM_KEY))
        if tel is None or sim is None:
            viewer.sync()
            time.sleep(0.01)
            continue

        if align is None:
            align = make_align_transform(
                (tel["root_xyz"], tel["root_quat"]),
                (sim["root_xyz"], sim["root_quat"]))
            now = time.time()
            if now - last_print > 1.0:
                print(f"[viz] aligned initial roots: "
                      f"teleop@{np.round(tel['root_xyz'], 3)} "
                      f"sim2sim@{np.round(sim['root_xyz'], 3)}")
                last_print = now

        data.qpos[TELEOP_SLICE] = state_to_qpos(tel)
        data.qpos[SIM2SIM_SLICE] = aligned_state_to_qpos(sim, align)
        mj.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.005)


def replay_mode(viewer, model, data, log_path, speed=1.0, loop=True):
    with open(log_path) as f:
        log = json.load(f)
    frames = log["frames"]
    if not frames:
        print("[viz] empty log.")
        return
    first = frames[0]
    align = make_align_transform(
        (first["teleop_state"]["root_xyz"], first["teleop_state"]["root_quat"]),
        (first["sim2sim_state"]["root_xyz"], first["sim2sim_state"]["root_quat"]))

    print(f"[viz] replay {len(frames)} frames, speed={speed}x, loop={loop}")
    last_t = None
    while viewer.is_running():
        for fr in frames:
            if not viewer.is_running():
                break
            data.qpos[TELEOP_SLICE] = state_to_qpos(fr["teleop_state"])
            data.qpos[SIM2SIM_SLICE] = aligned_state_to_qpos(fr["sim2sim_state"], align)
            mj.mj_forward(model, data)
            viewer.sync()
            if last_t is not None:
                dt_real = (fr["t_sim_ms"] - last_t) / 1000.0 / speed
                time.sleep(max(0.0, dt_real))
            last_t = fr["t_sim_ms"]
        if not loop:
            break
        print("[viz] replay finished, looping...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["live", "replay"], default="live")
    ap.add_argument("--log", default="logs/compare_run.json")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--redis_host", default="localhost")
    ap.add_argument("--redis_port", type=int, default=6379)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(DUAL_XML)
    data = mj.MjData(model)
    color_robots(model)

    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    with mjv.launch_passive(model=model, data=data,
                            show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        viewer.cam.lookat = np.array([0.0, 0.0, 0.9])
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        try:
            if args.mode == "live":
                live_mode(viewer, model, data, client)
            else:
                replay_mode(viewer, model, data, args.log,
                            speed=args.speed, loop=not args.no_loop)
        except KeyboardInterrupt:
            pass
    print("[viz] closed.")


if __name__ == "__main__":
    main()
