#!/usr/bin/env python3
"""
Teleop-vs-sim2sim robot state comparison recorder.

Records full robot states published by the teleop process
(compare_teleop_state) and the sim2sim process (compare_sim2sim_state) from
Redis, time-aligns the two streams, aligns the initial root frames, computes
per-frame differences, and writes:
  - <out>.csv    per-frame diffs (root pos/orientation, ankles, all 29 joints)
  - <out>.json   raw aligned samples + summary statistics

Usage (conda env: twist2):
  python compare_tools/compare_record.py --duration 30 --out logs/run1

Both producers must be running (teleop.sh and sim2sim.sh).
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

import numpy as np
import redis
from scipy.spatial.transform import Rotation as R

REDIS_TELEOP_KEY = "compare_teleop_state"
REDIS_SIM2SIM_KEY = "compare_sim2sim_state"

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
]


def quat_to_xyzw(wxyz):
    return np.asarray([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)


def xyzw_to_quat(xyzw):
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def quat_angle_deg(q1_wxyz, q2_wxyz):
    """Geodesic angle (deg) between two scalar-first quaternions."""
    r1 = R.from_quat(quat_to_xyzw(q1_wxyz))
    r2 = R.from_quat(quat_to_xyzw(q2_wxyz))
    rel = (r1.inv() * r2).as_rotvec()
    return float(np.degrees(np.linalg.norm(rel)))


def make_align_transform(teleop_root0, sim_root0):
    """Return a function that maps a sim2sim world pose (xyz, quat) into the
    teleop reference frame, using the first-frame root alignment:
    T = T_teleop(t0) * inv(T_sim2sim(t0))."""
    p_t = np.asarray(teleop_root0[0], dtype=float)
    q_t = R.from_quat(quat_to_xyzw(teleop_root0[1]))
    p_s = np.asarray(sim_root0[0], dtype=float)
    q_s = R.from_quat(quat_to_xyzw(sim_root0[1]))

    q_align = q_t * q_s.inv()

    def align(xyz, quat_wxyz):
        p = np.asarray(xyz, dtype=float)
        q = R.from_quat(quat_to_xyzw(quat_wxyz))
        p_new = q_align.apply(p - p_s) + p_t
        q_new = q_align * q
        return p_new, xyzw_to_quat(q_new.as_quat())

    return align


def parse_state(raw):
    if raw is None:
        return None
    st = json.loads(raw)
    required = ["t_ms", "root_xyz", "root_quat", "dof_pos",
                "ankle_left_xyz", "ankle_right_xyz"]
    for k in required:
        if k not in st:
            return None
    if len(st["dof_pos"]) != 29:
        return None
    st["t_ms"] = int(st["t_ms"])
    st["root_xyz"] = [float(x) for x in st["root_xyz"]]
    st["root_quat"] = [float(x) for x in st["root_quat"]]
    st["dof_pos"] = [float(x) for x in st["dof_pos"]]
    st["ankle_left_xyz"] = [float(x) for x in st["ankle_left_xyz"]]
    st["ankle_right_xyz"] = [float(x) for x in st["ankle_right_xyz"]]
    return st


class StreamBuffer:
    """Keeps the most recent samples per stream (dedup by t_ms)."""

    def __init__(self, maxlen=4096):
        self.samples = {}   # t_ms -> state
        self.maxlen = maxlen

    def add(self, st):
        if st is None:
            return
        self.samples[st["t_ms"]] = st
        if len(self.samples) > self.maxlen:
            oldest = sorted(self.samples)[: len(self.samples) - self.maxlen]
            for t in oldest:
                self.samples.pop(t, None)

    def sorted_items(self):
        return sorted(self.samples.items())


def match_pairs(teleop_buf, sim_buf, tol_ms=30.0):
    """Pair each sim2sim frame with the nearest teleop frame within tol_ms."""
    tel_items = teleop_buf.sorted_items()
    if not tel_items:
        return []
    tel_t = np.array([t for t, _ in tel_items])
    pairs = []
    for t_s, st_s in sim_buf.sorted_items():
        idx = int(np.argmin(np.abs(tel_t - t_s)))
        dt = abs(tel_t[idx] - t_s)
        if dt <= tol_ms:
            pairs.append((tel_items[idx][0], t_s, dt))
    return pairs


def compute_frame_diff(st_tel, st_sim, align):
    """Aligned per-frame difference dict."""
    p_sim_a, q_sim_a = align(st_sim["root_xyz"], st_sim["root_quat"])
    p_tel = np.asarray(st_tel["root_xyz"], dtype=float)

    ankle_l_sim_a, _ = align(st_sim["ankle_left_xyz"], [1, 0, 0, 0])
    ankle_r_sim_a, _ = align(st_sim["ankle_right_xyz"], [1, 0, 0, 0])
    ankle_l_tel = np.asarray(st_tel["ankle_left_xyz"], dtype=float)
    ankle_r_tel = np.asarray(st_tel["ankle_right_xyz"], dtype=float)

    dof_tel = np.asarray(st_tel["dof_pos"], dtype=float)
    dof_sim = np.asarray(st_sim["dof_pos"], dtype=float)

    return {
        "root_pos_err_m": float(np.linalg.norm(p_tel - p_sim_a)),
        "root_orient_err_deg": float(quat_angle_deg(st_tel["root_quat"], q_sim_a)),
        "ankle_left_err_m": float(np.linalg.norm(ankle_l_tel - ankle_l_sim_a)),
        "ankle_right_err_m": float(np.linalg.norm(ankle_r_tel - ankle_r_sim_a)),
        "joint_err_rad": [float(x) for x in (dof_tel - dof_sim)],
    }


def write_csv(path, frames):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["frame", "t_sim_ms", "t_tel_ms", "dt_ms",
                   "root_pos_err_m", "root_orient_err_deg",
                   "ankle_left_err_m", "ankle_right_err_m"]
                  + [f"j_{n}" for n in JOINT_NAMES])
        w.writerow(header)
        for fr in frames:
            w.writerow(
                [fr["frame"], fr["t_sim_ms"], fr["t_tel_ms"], fr["dt_ms"],
                 round(fr["root_pos_err_m"], 5), round(fr["root_orient_err_deg"], 4),
                 round(fr["ankle_left_err_m"], 5), round(fr["ankle_right_err_m"], 5)]
                + [round(x, 5) for x in fr["joint_err_rad"]]
            )


def summarize(frames):
    n = len(frames)
    if n == 0:
        return {}
    joint_err = np.array([fr["joint_err_rad"] for fr in frames])
    joint_abs = np.abs(joint_err)
    root_pos = np.array([fr["root_pos_err_m"] for fr in frames])
    root_orient = np.array([fr["root_orient_err_deg"] for fr in frames])
    ankle_l = np.array([fr["ankle_left_err_m"] for fr in frames])
    ankle_r = np.array([fr["ankle_right_err_m"] for fr in frames])

    joint_stats = []
    for i, name in enumerate(JOINT_NAMES):
        joint_stats.append({
            "joint": name,
            "leg": name.split("_")[0] in ("left", "right") and "hip" in name
                   or name.startswith(("left_", "right_")) and
                   any(k in name for k in ("knee", "ankle")),
            "rmse_rad": float(np.sqrt(np.mean(joint_abs[:, i] ** 2))),
            "max_rad": float(np.max(joint_abs[:, i])),
            "mean_rad": float(np.mean(joint_abs[:, i])),
        })
    joint_stats.sort(key=lambda s: -s["rmse_rad"])

    return {
        "n_frames": n,
        "root_pos_err_m": {"mean": float(np.mean(root_pos)),
                           "max": float(np.max(root_pos))},
        "root_orient_err_deg": {"mean": float(np.mean(root_orient)),
                                "max": float(np.max(root_orient))},
        "ankle_left_err_m": {"mean": float(np.mean(ankle_l)),
                             "max": float(np.max(ankle_l))},
        "ankle_right_err_m": {"mean": float(np.mean(ankle_r)),
                              "max": float(np.max(ankle_r))},
        "joint_stats_sorted": joint_stats,
    }


def print_summary(s):
    if not s:
        print("[compare] no matched frames.")
        return
    print("\n========== Compare Summary ==========")
    print(f"frames matched: {s['n_frames']}")
    print(f"root position err  : mean {s['root_pos_err_m']['mean']:.4f} m, "
          f"max {s['root_pos_err_m']['max']:.4f} m")
    print(f"root orientation   : mean {s['root_orient_err_deg']['mean']:.2f} deg, "
          f"max {s['root_orient_err_deg']['max']:.2f} deg")
    print(f"ankle L err        : mean {s['ankle_left_err_m']['mean']:.4f} m, "
          f"max {s['ankle_left_err_m']['max']:.4f} m")
    print(f"ankle R err        : mean {s['ankle_right_err_m']['mean']:.4f} m, "
          f"max {s['ankle_right_err_m']['max']:.4f} m")
    print("\njoint errors (deg), sorted by RMSE:")
    print(f"{'joint':24s} {'rmse_deg':>9s} {'mean_deg':>9s} {'max_deg':>9s}")
    for j in s["joint_stats_sorted"]:
        print(f"{j['joint']:24s} {np.degrees(j['rmse_rad']):9.2f} "
              f"{np.degrees(j['mean_rad']):9.2f} {np.degrees(j['max_rad']):9.2f}")
    print("=====================================")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="record duration in seconds (Ctrl+C stops earlier)")
    ap.add_argument("--out", default="logs/compare_run",
                    help="output path prefix (writes <out>.csv and <out>.json)")
    ap.add_argument("--tol_ms", type=float, default=30.0,
                    help="max time mismatch for pairing teleop/sim2sim frames")
    ap.add_argument("--redis_host", default="localhost")
    ap.add_argument("--redis_port", type=int, default=6379)
    args = ap.parse_args()

    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    client.ping()

    teleop_buf = StreamBuffer()
    sim_buf = StreamBuffer()
    stop = {"flag": False}

    def on_sigint(signum, frame):
        print("\n[compare] stopping...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_sigint)

    print(f"[compare] recording for {args.duration}s "
          f"(tol={args.tol_ms}ms). Press Ctrl+C to stop early.")
    t0 = time.time()
    poll_interval = 0.01
    while not stop["flag"] and time.time() - t0 < args.duration:
        tel = parse_state(client.get(REDIS_TELEOP_KEY))
        sim = parse_state(client.get(REDIS_SIM2SIM_KEY))
        teleop_buf.add(tel)
        sim_buf.add(sim)
        time.sleep(poll_interval)

    if not teleop_buf.samples or not sim_buf.samples:
        print("[compare] ERROR: no data received. Are teleop/sim2sim running "
              "with the state-publish patch?")
        sys.exit(1)

    pairs = match_pairs(teleop_buf, sim_buf, tol_ms=args.tol_ms)
    if len(pairs) < 2:
        print("[compare] ERROR: fewer than 2 matched frame pairs. "
              "Check that both processes are publishing fresh timestamps.")
        sys.exit(1)

    # Initial root alignment using the first matched pair.
    tel_first = teleop_buf.samples[pairs[0][0]]
    sim_first = sim_buf.samples[pairs[0][1]]
    align = make_align_transform(
        (tel_first["root_xyz"], tel_first["root_quat"]),
        (sim_first["root_xyz"], sim_first["root_quat"]))
    print(f"[compare] matched {len(pairs)} frame pairs. "
          f"Initial root alignment: teleop@{tel_first['root_xyz']} "
          f"sim2sim@{sim_first['root_xyz']}")

    frames = []
    for i, (t_tel, t_sim, dt) in enumerate(pairs):
        st_tel = teleop_buf.samples[t_tel]
        st_sim = sim_buf.samples[t_sim]
        diff = compute_frame_diff(st_tel, st_sim, align)
        frames.append({
            "frame": i,
            "t_tel_ms": int(t_tel),
            "t_sim_ms": int(t_sim),
            "dt_ms": round(float(dt), 3),
            **diff,
            "teleop_state": st_tel,
            "sim2sim_state": st_sim,
        })

    summary = summarize(frames)
    print_summary(summary)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_path = args.out + ".csv"
    json_path = args.out + ".json"
    write_csv(csv_path, frames)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "frames": frames}, f, indent=1)
    print(f"[compare] wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
