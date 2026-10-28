"""v2 reference deployment runtime: IMU + 16 SDK joints -> 570-dim history obs -> 16 joint commands at 50 Hz.

Runs on the MC board ("dog" / firefly, RK3588, PREEMPT_RT): numpy core, policy via **onnxruntime**
(installed there; 1.23.2) or TorchScript if torch is available. No Isaac, no project packages —
copy this file + joint_map_v2.py to the robot. Contract: OBS_CONTRACT_V2.md.

⚠ IMU QUATERNION IS (x, y, z, w) — MEASURED LIVE 2026-10-15, OVERRIDING the lowlevel.h comment ("w,x,y,z"):
    live sample  q = (-0.0049, 0.0016, -0.7956, 0.6058)  with  acc = (0.063, -0.306, 9.989)  (robot level)
    as xyzw: pure yaw -105°, projected gravity (-0.006, 0.008, -1.000)  -> matches acc  ✓
    as wxyz (header comment): gravity (0.006, 0.964, 0.266)             -> nonsense     ✗
    (scipy's as_quat() returns xyzw — the likely origin of the firmware's layout.)
The runtime therefore defaults to xyzw AND cross-checks quaternion-gravity against the accelerometer on the
first step after every reset (gate §7.12): standing still, -acc/|acc| must agree with the quaternion gravity
to within IMU_CHECK_MAX_DEG. Disagreement = wrong convention or firmware change -> REFUSE to run.

Two checkpoint PROFILES — the only difference is the wheel dispatch (contract §4.1):
    stage1: legs p_des = qd + scale*a (kp 20 / kd 0.7); wheels v_des=0, kp=0, kd=5.0 (DEPLOY-SIDE BRAKE,
            policy wheel outputs discarded — the policy does NOT lock the wheels itself)
    stage2: legs same; wheels v_des = 5.0 * a[12+k], kp=0, kd=0.7
    In BOTH, a_prev fed back into the observation is the policy's RAW 16-dim output.

Control step (50 Hz): read sensors -> SDK(4,4) reshape -> gravity(quat xyzw) ->
57-dim frame [gyro*0.25 | g | cmd | (q-qd, wheels 0) | qdot*0.05 | a_prev] (each term clip ±100 THEN scale)
-> 10-frame TERM-MAJOR history, oldest first (570) -> policy -> dispatch -> a_prev <- a (raw, last).
History reset REPLICATES the first frame (never zeros). Reset on power-up / mode switch / restart / sensor gap.

Self-test:  python3 loco_runtime_v2.py [--policy policy/xxx.onnx|xxx.ts.pt --profile stage1|stage2]
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_map_v2 import policy_to_sdk, sdk_to_policy  # noqa: E402

# ----------------------------------------------------------------------------------------- contract constants
HISTORY = 10
FRAME_DIM = 57
OBS_DIM = HISTORY * FRAME_DIM          # 570
N_ACT = 16
OBS_CLIP = 100.0
TERM_SLICES = (("ang_vel", 0, 3), ("gravity", 3, 6), ("cmd", 6, 9), ("q_rel", 9, 25), ("qdot", 25, 41), ("a_prev", 41, 57))
SCALE_ANG_VEL = 0.25
SCALE_QDOT = 0.05

Q_DEFAULT_LEGS = np.array((0.0, 0.8, -1.5) * 4, np.float32)         # rear hip 0.8 — v1's 1.0 is obsolete
ACTION_SCALE_LEGS = np.array((0.125, 0.25, 0.25) * 4, np.float32)   # per-joint! ABAD 0.125, HIP/KNEE 0.25
LEG_KP, LEG_KD = 20.0, 0.7
WHEEL_SCALE, WHEEL_KD_ROLL, WHEEL_KD_BRAKE = 5.0, 0.7, 5.0

QUAT_LAYOUT = "xyzw"                   # measured live; see module docstring
IMU_CHECK_MAX_DEG = 15.0               # first-step gravity-vs-accelerometer agreement gate
PROFILES = ("stage1", "stage2")

# regression fixture: the live sample that settled the convention (robot level, pure yaw)
_LIVE_Q_XYZW = (-0.0049, 0.0016, -0.7956, 0.6058)
_LIVE_ACC = (0.063, -0.306, 9.989)


def gravity_from_quat(q4, layout: str = QUAT_LAYOUT) -> np.ndarray:
    """Projected gravity UNIT vector in the base frame: R(q)^T @ (0,0,-1). Normalizes q first."""
    q = np.asarray(q4, np.float64).reshape(4)
    if layout == "xyzw":
        x, y, z, w = q
    elif layout == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(layout)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-8:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [2 * w * y - 2 * x * z, -2 * w * x - 2 * y * z, -(2 * w * w - 1) - 2 * z * z], np.float32
    )


def gravity_acc_angle_deg(quat4, acc3, layout: str = QUAT_LAYOUT) -> float:
    """Angle between quaternion-gravity and -acc/|acc| (valid when stationary). Small = convention OK."""
    g = gravity_from_quat(quat4, layout)
    a = -np.asarray(acc3, np.float64).reshape(3)
    a = a / max(np.linalg.norm(a), 1e-8)
    cos = float(np.clip(np.dot(g, a), -1.0, 1.0))
    return math.degrees(math.acos(cos))


class HistoryBuffer:
    """(10, 57) frame store -> 570-dim TERM-MAJOR observation, oldest frame first within each term."""

    def __init__(self):
        self.frames = np.zeros((HISTORY, FRAME_DIM), np.float32)
        self.filled = False

    def reset(self) -> None:
        self.filled = False

    def push(self, frame: np.ndarray) -> np.ndarray:
        f = np.asarray(frame, np.float32).reshape(FRAME_DIM)
        if not self.filled:
            self.frames[:] = f            # replicate-fill: "has been holding this state", never zeros
            self.filled = True
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = f
        return np.concatenate([self.frames[:, s:e].reshape(-1) for (_, s, e) in TERM_SLICES])


# ------------------------------------------------------------------ policy backends (ONNX preferred on the dog)
class _OrtPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.in_name: obs.reshape(1, OBS_DIM).astype(np.float32)})[0]
        return np.asarray(out, np.float32).reshape(-1)


class _TorchPolicy:
    def __init__(self, path: str):
        import torch

        self.torch = torch
        try:
            self.mod = torch.jit.load(path, map_location="cpu")
        except Exception:
            ck = torch.load(path, map_location="cpu")
            state = ck["model_state_dict"] if "model_state_dict" in ck else ck
            import torch.nn as nn

            keys = sorted([k for k in state if k.startswith("actor.") and k.endswith(".weight")],
                          key=lambda k: int(k.split(".")[1]))
            layers = []
            for i, k in enumerate(keys):
                w, b = state[k], state[k.replace(".weight", ".bias")]
                lin = nn.Linear(w.shape[1], w.shape[0])
                lin.weight.data.copy_(w)
                lin.bias.data.copy_(b)
                layers.append(lin)
                if i < len(keys) - 1:
                    layers.append(nn.ELU())
            self.mod = nn.Sequential(*layers)
        self.mod.eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.from_numpy(obs.reshape(1, OBS_DIM).astype(np.float32))
            return self.mod(t).numpy().reshape(-1)


def load_policy(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.endswith(".onnx"):
        return _OrtPolicy(path)
    return _TorchPolicy(path)


class LocoRuntimeV2:
    def __init__(self, policy_path: str, profile: str = "stage1"):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
        self.profile = profile
        self.policy = load_policy(policy_path) if policy_path else None
        self.history = HistoryBuffer()
        self.a_prev = np.zeros(N_ACT, np.float32)
        self._imu_check_pending = True

    def reset(self) -> None:
        """Power-up / mode switch / controller restart / sensor gap. Call BEFORE the first step after any of those."""
        self.a_prev[:] = 0.0
        self.history.reset()
        self._imu_check_pending = True

    @staticmethod
    def _clip(x):
        return np.clip(np.asarray(x, np.float32), -OBS_CLIP, OBS_CLIP)

    def step(self, gyro, quat, q16_sdk, qdot16_sdk, cmd, acc=None):
        """One 50 Hz step. quat is the raw 4 floats from imu_shm (xyzw — see module docstring).

        Pass ``acc`` (raw m/s², from imu_shm) at least on the FIRST step after reset(): the runtime
        cross-checks the quaternion gravity against -acc and REFUSES to run on disagreement (§7.12).
        Returns SDK-order lists (len 16): p_des, v_des, kp, kd, t_ff — hand them to the >=500 Hz
        refresh thread (contract §5) — plus 'action' (raw 16) and 'obs' (570, np.float32).
        """
        if self.policy is None:
            raise RuntimeError("no policy loaded")
        grav = gravity_from_quat(quat)
        if self._imu_check_pending:
            if acc is None:
                raise RuntimeError("pass acc= on the first step after reset(): IMU convention gate (§7.12)")
            ang = gravity_acc_angle_deg(quat, acc)
            if ang > IMU_CHECK_MAX_DEG:
                raise RuntimeError(
                    f"IMU convention gate FAILED: quaternion gravity vs -acc disagree by {ang:.1f}° "
                    f"(> {IMU_CHECK_MAX_DEG}°). Wrong quat layout or firmware change — DO NOT hand over."
                )
            self._imu_check_pending = False

        q_legs, _q_wheels = sdk_to_policy(list(np.asarray(q16_sdk, np.float32).reshape(16)))
        qd_legs, qd_wheels = sdk_to_policy(list(np.asarray(qdot16_sdk, np.float32).reshape(16)))
        q_legs = np.asarray(q_legs, np.float32)
        qdot_pol = np.asarray(qd_legs + qd_wheels, np.float32)

        q_rel16 = np.zeros(16, np.float32)
        q_rel16[:12] = q_legs - Q_DEFAULT_LEGS            # wheel positions stay 0, like training

        frame = np.concatenate([
            self._clip(gyro).reshape(3) * SCALE_ANG_VEL,
            self._clip(grav).reshape(3),
            self._clip(cmd).reshape(3),
            self._clip(q_rel16),
            self._clip(qdot_pol) * SCALE_QDOT,
            self._clip(self.a_prev),
        ])
        obs = self.history.push(frame)

        a = np.asarray(self.policy(obs), np.float32).reshape(N_ACT)

        leg_p = (Q_DEFAULT_LEGS + ACTION_SCALE_LEGS * a[:12]).tolist()
        if self.profile == "stage1":
            wheel_v, wheel_kd = [0.0] * 4, [WHEEL_KD_BRAKE] * 4       # deploy-side brake; a[12:16] discarded
        else:
            wheel_v, wheel_kd = (WHEEL_SCALE * a[12:16]).tolist(), [WHEEL_KD_ROLL] * 4

        out = {
            "p_des": policy_to_sdk(leg_p, [0.0] * 4),
            "v_des": policy_to_sdk([0.0] * 12, wheel_v),
            "kp": policy_to_sdk([LEG_KP] * 12, [0.0] * 4),
            "kd": policy_to_sdk([LEG_KD] * 12, wheel_kd),
            "t_ff": [0.0] * 16,
            "action": a.tolist(),
            "obs": obs,
        }
        self.a_prev = a.copy()                            # raw output, updated LAST (contract §4 step 8)
        return out


# ------------------------------------------------------------------ self-test (contract §7, software half)
def _selftest(policy_path: str, profile: str) -> int:
    bad = 0

    def check(name, ok, detail=""):
        nonlocal bad
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        bad += 0 if ok else 1

    hb = HistoryBuffer()
    f0 = np.arange(FRAME_DIM, dtype=np.float32)
    obs = hb.push(f0)
    check("§7.1 obs width 570 / frame 57 / H 10", obs.size == OBS_DIM)
    check("§7.2 reset replicates the first frame into all 10 slots", bool((hb.frames == f0).all()))

    hb.reset()
    f = np.zeros(FRAME_DIM, np.float32)
    for k, (_, s, e) in enumerate(TERM_SLICES, start=1):
        f[s:e] = float(k)
    obs = hb.push(f)
    off, ok = 0, True
    for k, (_, s, e) in enumerate(TERM_SLICES, start=1):
        seg = obs[off: off + HISTORY * (e - s)]
        ok &= bool((seg == float(k)).all())
        off += HISTORY * (e - s)
    check("§7.3 term-major layout (6 segments of 10 frames each)", ok)

    hb.reset()
    for t in range(HISTORY):
        obs = hb.push(np.full(FRAME_DIM, float(t), np.float32))
    check("§7.4 oldest frame first within each term",
          bool((obs[:HISTORY * 3].reshape(HISTORY, 3)[:, 0] == np.arange(10)).all()))

    def _mk(policy, prof):
        rt = LocoRuntimeV2.__new__(LocoRuntimeV2)
        rt.profile, rt.policy = prof, policy
        rt.history, rt.a_prev = HistoryBuffer(), np.zeros(N_ACT, np.float32)
        rt._imu_check_pending = True
        return rt

    level_q, level_acc = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 9.81)      # xyzw identity + level acc
    rt = _mk(lambda o: np.ones(N_ACT, np.float32), profile)
    out = rt.step([0, 0, 0], level_q, policy_to_sdk(list(Q_DEFAULT_LEGS), [0] * 4), [0.0] * 16, [0, 0, 0], acc=level_acc)
    legs, _ = sdk_to_policy(out["p_des"])
    delta = [legs[i] - Q_DEFAULT_LEGS[i] for i in range(12)]
    check("§7.5 per-joint action scale (a=1 -> ABAD +0.125, HIP/KNEE +0.25)",
          all(abs(delta[i] - ACTION_SCALE_LEGS[i]) < 1e-6 for i in range(12)))
    vk, kk = sdk_to_policy(out["v_des"])[1], sdk_to_policy(out["kd"])[1]
    if profile == "stage1":
        check("§7.6 stage1 wheels: v_des=0, kp=0, kd=5.0 (brake), a[12:16] discarded",
              all(v == 0 for v in vk) and all(abs(k - WHEEL_KD_BRAKE) < 1e-6 for k in kk))
    else:
        check("§7.6 stage2 wheels: v_des=5*a, kd=0.7",
              all(abs(v - WHEEL_SCALE) < 1e-6 for v in vk) and all(abs(k - WHEEL_KD_ROLL) < 1e-6 for k in kk))

    probe = list(range(100, 116))
    l, w = sdk_to_policy(probe)
    check("§7.7 SDK<->policy reshape roundtrip", policy_to_sdk(l, w) == probe)

    # §7.12 IMU convention: the LIVE sample from 2026-10-15 is the regression fixture
    ang_xyzw = gravity_acc_angle_deg(_LIVE_Q_XYZW, _LIVE_ACC, "xyzw")
    ang_wxyz = gravity_acc_angle_deg(_LIVE_Q_XYZW, _LIVE_ACC, "wxyz")
    check("§7.12 live IMU sample: xyzw agrees with acc (<3°), wxyz does not (>45°)",
          ang_xyzw < 3.0 and ang_wxyz > 45.0, f"xyzw {ang_xyzw:.2f}°, wxyz {ang_wxyz:.1f}°")
    try:
        _mk(lambda o: np.zeros(N_ACT, np.float32), profile).step(
            [0, 0, 0], (0.6058, -0.0049, 0.0016, -0.7956),  # the live quat DELIBERATELY fed as if wxyz-ordered
            policy_to_sdk(list(Q_DEFAULT_LEGS), [0] * 4), [0.0] * 16, [0, 0, 0], acc=_LIVE_ACC)
        check("§7.12b wrong-layout quat is REFUSED at first step", False)
    except RuntimeError:
        check("§7.12b wrong-layout quat is REFUSED at first step", True)

    if policy_path:
        rt = _mk(load_policy(policy_path), profile)
        rt.reset()
        q_sdk = policy_to_sdk(list(Q_DEFAULT_LEGS), [0.0] * 4)
        out = None
        for t in range(200):
            out = rt.step([0, 0, 0], level_q, q_sdk, [0.0] * 16, [0, 0, 0], acc=level_acc)
        legs, _ = sdk_to_policy(out["p_des"])
        a = np.asarray(out["action"])
        abad_t = [abs(legs[i] - Q_DEFAULT_LEGS[i]) for i in (0, 3, 6, 9)]
        knee_t = [legs[i] for i in (2, 5, 8, 11)]
        check("§7.8a outputs finite after 200 standing steps", bool(np.isfinite(a).all()))
        check("§7.8b ABAD targets within soft limit 0.4398", max(abad_t) <= 0.4398, f"max {max(abad_t):.4f}")
        check("§7.8c knee targets within hard limits [-2.723, -0.602]",
              all(-2.723 <= k <= -0.602 for k in knee_t), f"{['%.3f' % k for k in knee_t]}")
        print(f"[info] standing fixed point: max|a_leg| {float(np.abs(a[:12]).max()):.3f} "
              f"(gravity compensation ~1.2-1.5 on knees is EXPECTED, not a fault — ledger §2.1), "
              f"max|a_wheel| {float(np.abs(a[12:16]).max()):.3f}"
              f" ({'discarded' if profile == 'stage1' else 'v_des %.2f rad/s' % float(5 * np.abs(a[12:16]).max())})")

    print("[selftest]", "OK" if bad == 0 else f"{bad} FAILED")
    return bad


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="", help="policy/*.onnx (dog) or *.ts.pt / model_*.pt (needs torch)")
    ap.add_argument("--profile", type=str, default="stage1", choices=PROFILES)
    args = ap.parse_args()
    raise SystemExit(1 if _selftest(args.policy, args.profile) else 0)
