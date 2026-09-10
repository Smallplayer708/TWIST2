#!/usr/bin/env python3
"""TWIST2 Teleop Diagnostic Monitor
Monitors Redis data flow from gmr teleop to low-level controller.
Detects state transitions and arm motion in real-time.
Usage: python diag_monitor.py [--env gmr|twist2]
Output: stdout + twist2_diag.log
"""

import redis, json, time, sys, os, argparse
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "twist2_diag.log")
REDIS_KEYS = {
    "body": "action_body_unitree_g1_with_hands",
    "state": "state_body_unitree_g1_with_hands",
    "hand_l": "action_hand_left_unitree_g1_with_hands",
    "hand_r": "action_hand_right_unitree_g1_with_hands",
    "neck": "action_neck_unitree_g1_with_hands",
    "ctrl": "controller_data",
}

DEFAULT_ARMS = [0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
                0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0]

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except IOError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["gmr","twist2"], default="any",
                       help="Which env this monitor runs in (for log context)")
    args = parser.parse_args()

    log(f"=== DIAG START (env={args.env}) ===")

    # 1. Redis check
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=3)
        r.ping()
        log(f"  [OK] Redis connected, {r.dbsize()} keys in DB 0")
    except Exception as e:
        log(f"  [FAIL] Redis: {e}")
        log("=== DIAG END (FAIL) ===")
        return

    # 2. Check if gmr key exists
    if r.get(REDIS_KEYS["body"]) is None:
        log(f"  [WARN] Key '{REDIS_KEYS['body']}' not found")
        log("  → Start gmr teleop FIRST, then run this monitor")
        log("=== DIAG END (no data) ===")
        return

    log(f"  Monitoring '{REDIS_KEYS['body']}' for arm motion...")
    log(f"  Default arms: {DEFAULT_ARMS}")
    log("")

    state = "startup"
    last_arms = None
    last_ctrl_state = None
    stationary_count = 0
    iteration = 0

    try:
        while True:
            iteration += 1
            body_raw = r.get(REDIS_KEYS["body"])
            ctrl_raw = r.get(REDIS_KEYS["ctrl"])

            if body_raw is None:
                if state != "no_data":
                    state = "no_data"
                    log(f"[{iteration:05d}] STATE=NO_DATA — gmr teleop stopped?")
                time.sleep(1)
                continue

            try:
                body = json.loads(body_raw)
                arms = body[21:35]
            except (json.JSONDecodeError, IndexError):
                log(f"[{iteration:05d}] [ERROR] Invalid body data")
                time.sleep(1)
                continue

            # --- State detection ---
            if last_arms is not None:
                arm_diff = sum(abs(a - b) for a, b in zip(arms, last_arms))
                if arm_diff > 0.005:
                    if state != "TELEOP_ACTIVE":
                        state = "TELEOP_ACTIVE"
                        stationary_count = 0
                        log(f"[{iteration:05d}] STATE=TELEOP_ACTIVE  arms={[round(v,3) for v in arms]}")
                    stationary_count = 0
                else:
                    stationary_count += 1
                    if state == "TELEOP_ACTIVE" and stationary_count >= 40:
                        state = "IDLE_PAUSE"
                        log(f"[{iteration:05d}] STATE=IDLE_PAUSE  stationary for {stationary_count} frames")
            else:
                # First reading
                def_diff = sum(abs(a - b) for a, b in zip(arms, DEFAULT_ARMS))
                if def_diff < 0.01:
                    state = "IDLE_PAUSE"
                    log(f"[{iteration:05d}] STATE=IDLE_PAUSE  arms match default")
                else:
                    state = "TELEOP_ACTIVE"
                    log(f"[{iteration:05d}] STATE=TELEOP_ACTIVE  arms={[round(v,3) for v in arms]}")

            last_arms = arms

            # --- Controller state ---
            if ctrl_raw:
                try:
                    ctrl = json.loads(ctrl_raw)
                    rc = ctrl.get("RightController", {})
                    lc = ctrl.get("LeftController", {})
                    ctrl_state = (
                        rc.get("key_one", False),
                        lc.get("key_one", False),
                        rc.get("index_trig", 0.0) > 0.5,
                        lc.get("index_trig", 0.0) > 0.5,
                    )
                    if ctrl_state != last_ctrl_state:
                        log(f"[{iteration:05d}] CTRL  A={ctrl_state[0]} X={ctrl_state[1]}  "
                              f"Rtrig={ctrl_state[2]} Ltrig={ctrl_state[3]}")
                        last_ctrl_state = ctrl_state
                except Exception:
                    pass

            time.sleep(0.5)

    except KeyboardInterrupt:
        log(f"\n[{iteration:05d}] === DIAG END (user interrupt) ===")
    except Exception as e:
        log(f"[{iteration:05d}] [ERROR] {e}")
        log("=== DIAG END (error) ===")

if __name__ == "__main__":
    main()
