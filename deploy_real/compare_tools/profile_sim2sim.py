#!/usr/bin/env python3
"""Profile the sim2sim loop stages to find the 30Hz bottleneck."""
import json
import os
import sys
import time

import mujoco
import mujoco.viewer as mjv
import numpy as np
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server_low_level_g1_sim import load_onnx_policy

XML = "/home/user/TWIST2/assets/g1/g1_sim2sim_29dof.xml"
CKPT = "/home/user/TWIST2/assets/ckpts/twist2_1017_20k.onnx"

model = mujoco.MjModel.from_xml_path(XML)
model.opt.timestep = 0.001
data = mujoco.MjData(model)

client = redis.Redis(host="localhost", port=6379, db=0)
pipe = client.pipeline()

for device in ("cpu", "cuda"):
    try:
        policy = load_onnx_policy(CKPT, device)
    except Exception as e:
        print(f"[{device}] load policy failed: {e}")
        continue
    obs = np.random.randn(1432).astype(np.float32)
    obs_t = __import__("torch").from_numpy(obs).float().unsqueeze(0).to(device)
    for _ in range(3):
        with __import__("torch").no_grad():
            policy(obs_t)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        with __import__("torch").no_grad():
            policy(obs_t)
    dt_policy = (time.perf_counter() - t0) / n * 1000
    print(f"[{device}] ONNX inference: {dt_policy:.2f} ms")

# mj_step cost
t0 = time.perf_counter()
n = 5000
for _ in range(n):
    mujoco.mj_step(model, data)
dt_step = (time.perf_counter() - t0) / n * 1000
print(f"mj_step: {dt_step:.3f} ms")

# viewer sync cost
viewer = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
for _ in range(5):
    viewer.sync()
t0 = time.perf_counter()
n = 300
for _ in range(n):
    mujoco.mj_step(model, data)
    viewer.sync()
dt_sync = (time.perf_counter() - t0) / n * 1000
print(f"mj_step+viewer.sync: {dt_sync:.3f} ms")
viewer.close()

# redis pipeline round trip (set+get batch like the main loop)
def redis_roundtrip():
    state = {"t_ms": int(time.time() * 1000),
             "root_xyz": [0.0, 0.0, 0.8], "root_quat": [1, 0, 0, 0],
             "dof_pos": [0.0] * 29,
             "ankle_left_xyz": [0, 0, 0], "ankle_right_xyz": [0, 0, 0]}
    pipe.set("compare_sim2sim_state", json.dumps(state))
    pipe.set("state_body_unitree_g1_with_hands", json.dumps([0.0] * 34))
    pipe.set("t_state", int(time.time() * 1000))
    pipe.execute()
    for k in ["action_body_unitree_g1_with_hands", "action_hand_left_unitree_g1_with_hands",
              "action_hand_right_unitree_g1_with_hands", "action_neck_unitree_g1_with_hands"]:
        pipe.get(k)
    res = pipe.execute()
    json.loads(res[0])

for _ in range(10):
    redis_roundtrip()
t0 = time.perf_counter()
n = 500
for _ in range(n):
    redis_roundtrip()
dt_redis = (time.perf_counter() - t0) / n * 1000
print(f"redis roundtrip (set-batch + get-batch + json): {dt_redis:.3f} ms")

# expected policy cycle with decimation=20 (50Hz target):
print("\ncycle budget @50Hz target (decimation=20, sim_dt=0.001):")
print(f"  19 x mj_step (1ms each via limit_fps) = 19.0 ms")
print(f"  1 x policy step = mj_step + redis + inference + viewer.sync")
