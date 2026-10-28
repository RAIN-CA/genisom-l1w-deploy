#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GENISOM L1-W v2 main control loop — 570-dim history-observation policy -> spline_shm (runs on the dog / firefly RK3588).

Default is **dry-run**: the full chain executes but never writes cmd (mc_ctrl may keep running, zero risk).
Add --live to actually take over (mc_ctrl MUST be stopped first). Contract: OBS_CONTRACT_V2.md. This file
does I/O + state machine only; all observation/action math lives in loco_runtime_v2.LocoRuntimeV2
(single source of truth — do not re-implement it here).

    # On-robot software gates (no shared-memory access; run BOTH profiles; all PASS before --live is allowed):
    python3 runtime/run_loco_v2.py --check --policy policy/policy_stage1_model_4999.onnx --profile stage1
    # dry-run (reads real sensors, writes no cmd; also measures the actual sender rate on this machine):
    python3 runtime/run_loco_v2.py --policy policy/policy_stage1_model_4999.onnx --profile stage1
    # live (stop mc_ctrl first; robot on the hoist; e-stop within reach):
    sudo python3 runtime/run_loco_v2.py --live --policy policy/policy_stage1_model_4999.onnx --profile stage1
    # gate #7 per-joint jog (hoisted) / gate #10 per-wheel raw spin-direction mapping (hoisted):
    sudo python3 runtime/run_loco_v2.py --live --jog        --policy ... --profile stage1
    sudo python3 runtime/run_loco_v2.py --live --wheel_test --policy ... --profile stage1

Keys (r3, operator-specified WASD layout; stand-up / enter-policy moved to number keys so steering
can't accidentally switch state):
    1 stand up    2 enter policy    SPACE e-stop (back to damping)    c zero the command    r save dump now
    hold w/s forward/backward  a/d strafe left/right  q/e turn left/right
    (release for <=0.6 s auto-zeros -> stand in place; --step_keys switches to stepped increments)

Structure (contract §5/§6):
    sender thread 500 Hz: writes the current 16 joint targets to spline_shm + CONSUMER_CONTROL + leg.flags=1
                          (the 1 ms SPLINE daemon clears motor commands after ~10 cycles without a refresh,
                          so writing directly at 50 Hz WOULD trip it)
    main thread    50 Hz: DAMPING -> STAND (r15: replay of the recorded factory stand-up; falls back to a
                          2 s ramp when no trajectory file) -> HOLD -> POLICY; manual key gate per level,
                          no level skipping
    e-stops: SPACE in any state returns to DAMPING; tip-over (projG_z > -0.5, i.e. ~60° tilt — stricter than
             the contract's |r/p|>1.6), torque (live), suspected IMU stream stall, and policy targets older
             than 100 ms (sender watchdog) all drop to damping automatically

Deliberate differences from v1 (these v1 behaviors are FORBIDDEN here — see ledger/audit):
    legs kp20/kd0.7 (v1 30/0.6); no gravity-sag precompensation (v1 SAG_*); no target blending (closed loop
    immediately); wheels stage1 = v_des=0, kp=0, kd=5.0 velocity brake (v1 held position kp2/kd0.5 — different
    semantics), stage2 = v_des=5*a[12:16], kd=0.7; a_prev=0 straight into closed loop (v1 tried a frozen-obs
    warm-up once — it converges to a fixed point at 2.4x action).

Reused v1 parts that are already validated on this robot: spline_shm/imu_shm layout and access,
raw<->urdf sign map _S/_C, 500 Hz sender + watchdog, RT scheduling (FIFO 40 @ cores 5,6 — NEVER core 7),
the LIVE confirmation and key handling.

⚠ Wheel spin signs (WHEEL_SIGN) are NOT yet measured (gate #10). stage1 brakes the wheels (v=0), which is
   sign-independent, so it may go first; **before any stage2 run you MUST run --wheel_test, cross-check
   against simulation, and fill WHEEL_SIGN in — stage2 is forbidden until then.**

r2 (2026-10-15, after the dry-run): disk writes moved to a background thread; burst threshold |a|>8 or
tau>26 with a 10 s cooldown; live policy entry gained a pose gate (max leg |q-q_default| <= 0.2 rad — refuse
the handover if STAND did not truly arrive); telemetry gained aL.
r3 (2026-10-15, operator request): keys remapped to w/s fwd/back, a/d strafe, q/e turn, stand=1, policy=2,
e-stop=SPACE; key-release auto-zero window 180 -> 600 ms (a terminal's first auto-repeat delay is ~500 ms;
180 ms made held keys hiccup at the start).
r15 (2026-10-16, ledger §5.10): STAND replaced by a **replay of the recorded factory stand-up**
(logs/factory_traj_standup.npz, recorded with --record; blend-in 1.2 s -> replay at original speed ->
blend-out 0.5 s to q_default). Why: a straight ramp makes the rear legs push from a splayed pose (terrible
leverage) + the machine is rear-heavy, and kp60 has no gravity compensation -> the rear end sat down and
collapsed; the factory sequence is "tuck the feet under the body first, then push up". Validated in
simulation first (replay_standup_sim.py: level rise, max|pitch| 2.5°, final 0.345 m / -2.4°). Details:
wheels run a soft damping kd0.5 during the replay (they passively roll 4-6 rad in the factory stand-up;
braking them fights the tuck), the profile brake returns from the blend-out onward; pressing 1 while
already near the stance (<=0.35 rad) skips the replay and fine-trims directly; with no trajectory file it
falls back to the straight ramp with a warning. Torque e-stop during the replay has hysteresis: instant
stop above 32, 27~32 must persist 100 ms (the kp60 PD transient of the fast-swing segment can graze 27;
factory recording peaks at ~10, standing 6~9). HOLD/POLICY kept the instant 27 line (until r18).
r16 (2026-10-16, after the first successful on-robot replay stand-up): ① steering speeds raised to half the
training ranges (±2.0/±0.8/±1.5) = 1.0/0.4/0.75 (0.3/0.15/0.3 felt unresponsive per the operator);
② the stand-up "clicking" noise: the STAND segment now sends **500 Hz linearly smoothed targets**
(previously a 50 Hz staircase; in the fast swing each tick jumped 0.075 rad x kp60 ≈ 4.5 N*m of torque
ripple — the prime suspect for the faint clicks; smoothing divides the ripple by 10). POLICY stays 50 Hz
zero-order hold — that is what training saw, do not touch it. Limit audit (from the recording): our replay
targets stay clamped >=1e-3 inside the URDF limits the whole way; the factory itself parks the abad ON the
mechanical stop in the prone pose (measured 0.005~0.010 rad beyond the URDF limit) and the knee 0.007 from
its stop — resting on the stop is factory behavior, and our clamped targets press in with <=0.6 N*m, so it
is not the noise source.
r17 (2026-10-16, after the first walk ended in a fall caused by a 29.7 N*m false trip; verdict from the
e-stop ring-buffer data): torque-trip hysteresis **extended to every live state** (was replay-only).
Reconstruction: walking at 1.0 m/s was normal (per-frame torque peaks p50 18.4 / p90 20.3 / max 22.1,
attitude level throughout, no stumble precursor); the operator pressed 'a', the command jumped
(1,0,0)->(0,0.4,0) — brake-plus-strafe is a trained transition behavior — and a 29.7 N*m stance-leg
transient touched the instant-27 line -> e-stop cut to full-body damping mid-stride -> the fall.
**The e-stop caused the fall; it was not a consequence of one.** The r17 rule (instant >32; 27~32 for
100 ms) still false-tripped — a second e-stop proved that **zero-command standing** also grazes 27~30 on
the front hips (FL_hip 28.9 / FR_hip 19.3), not just transitions. r18 switched to motor-protection
criteria (below). Automatic e-stops also write estop_*.json (trigger-frame why/state/tau16/qd16/grav/cmd —
the ring buffer does not contain the trigger tick; the json captures it).

r18 (2026-10-16, after the second false trip): the torque e-stop keeps only two rules, matching the motor's
real failure modes — impact and sustained overload:
① max|tau| > tau_impact (default 40; peak rating 48, driver self-limits) -> stop immediately;
② a single joint with |tau| > tau_overload (default 32) for 600 ms CONTINUOUS -> stop (walking/standing
   transients and static loads at 28~30 all pass). New CLI --tau_overload / --tau_impact. If the policy
   rides the motor ceiling (e.g. ~29 on the front hips while standing), that is a signal to retrain with a
   torque penalty, not something the deploy side should absorb — see ledger §5.14 for that verdict.

r19 (2026-10-17, first on-robot video deploy1.mp4 t≈15.3-16.6 s: sprint at 1 m/s, release -> body pitches
forward, rear lightens, near forward-flip, recovers after ~1 s): **command slew**. The deploy-side command
was a step (release snaps to zero after 600 ms; direction keys replace instantly), while training's
transition reward (0.5 s window) explicitly rewards hard braking — so the robot executes its learned hard
brake, and the real machine pitches much deeper than sim. (Ledger §5.16/§5.20/§5.23: the top-mounted nav
kit is INSIDE the trained mass/CoM randomization — add_base_mass (-1,+4) kg and CoM z (0,+0.08) m train
from iteration 1 — and the sim-side friction/payload sweeps do NOT reproduce the flip; the residual gap is
actuator saturation transients / wheel-contact modeling / 50 Hz ZOH+latency. So the flip's deploy-side fix
is this slew; the training side is closed on this topic.) Mechanism: the operator target cmd and the
applied cmd are separated; the applied value slides toward the target under an acceleration limit
(default vx/vy 2.0 m/s², wz 4.0 rad/s²; a 1.0 m/s brake takes 0.5 s). Tune with --cmd_slew (0=off=old step).

On-robot --check pre-registered gates (C1-C8; all must PASS; C8 is skipped with a notice when the
trajectory file is absent):
    C1 raw<->urdf round trip (random + q_default)      C2 200 ideal standing steps: finite, ABAD<=0.4398, knees within hard limits
    C3 wheel dispatch per profile (stage1 brake / stage2 velocity)    C4 leg dispatch kp20/kd0.7 and limit clamp effective
    C5 344-byte cmd buffer field-by-field round trip + flags/CF       C6 STAND interpolation <=0.05 rad per tick, endpoint = q_default
    C7 dump.npy shape (N,586) with [0:570]=obs [570:586]=act (probe section-E compatible)
    C8 stand-up replay trajectory usable: duration 5~30 s, final frame within 0.3 of q_default, peak speed <6 rad/s, inside limits throughout
"""
from __future__ import annotations

import argparse
import gc
import mmap
import os
import signal
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_map_v2 import (POLICY_LEG_JOINT_NAMES, POLICY_WHEEL_NAMES,  # noqa: E402
                          SDK_JOINT_NAMES, WHEEL_SDK_INDICES, policy_to_sdk, sdk_to_policy)
from loco_runtime_v2 import (ACTION_SCALE_LEGS, LEG_KD, LEG_KP, N_ACT, OBS_DIM,  # noqa: E402
                             Q_DEFAULT_LEGS, WHEEL_KD_BRAKE, WHEEL_KD_ROLL, WHEEL_SCALE,
                             LocoRuntimeV2, gravity_from_quat)

# ═════════════════════════ configuration (v2 contract values; v1 field-proven run parameters) ═════════════════════════
POLICY_HZ, SEND_HZ = 50.0, 500.0
STAND_KP, STAND_KD = 60.0, 5.0        # contract §6: high-gain PD for STAND
STANDUP_SEC        = 2.0
STAND_STEP_MAX     = 0.05             # contract §6: homing moves <=0.05 rad per tick
DAMP_KD            = 3.0              # v1 field value
TORQUE_TRIP        = 27.0             # obsolete (r17 instant-hysteresis line); r18 uses TAU_OVERLOAD/TAU_IMPACT
FALL_GRAV_Z        = -0.5             # projG_z above this = tip-over (~60°, stricter than the contract's 1.6 rad)
WATCHDOG_MS        = 100.0            # stale policy targets -> sender drops gains to damping (contract #11: keep refreshing)
IMU_STALE_TICKS    = 25               # N consecutive 50 Hz ticks with ts AND gyro bit-identical (=500 ms) -> stream stall
RT_CORES, RT_PRIO  = (5, 6), 40       # NEVER include core 7: spline_daemon/mc_ctrl/imu_daemon live there
MAX_VX, MAX_VY, MAX_WZ = 1.0, 0.4, 0.75         # r16: half of the training ranges (±2.0, ±0.8, ±1.5)
CMD_STEP           = 0.1
HOLD_MODE          = True             # move only while held, stop on release (--step_keys disables)
HOLD_RELEASE_MS    = 600.0            # r3: > a terminal's first auto-repeat delay (~500 ms) or held keys hiccup
HOLD_VX, HOLD_VY, HOLD_WZ = 1.0, 0.4, 0.75      # r16: same; the old 0.3/0.15/0.3 felt unresponsive (operator)
DUMP_FRAMES        = 600              # contract §9: 600 frames of [obs570|act16] auto-saved
RING_FRAMES        = 200              # window kept around an anomaly trigger
ANOM_A, ANOM_TAU   = 8.0, 26.0        # burst thresholds (r2: prone pose reached |a|=10 in the dry-run; 3.0 spammed)
BURST_COOLDOWN_S   = 10.0             # minimum gap between burst saves (r2: stop trigger chains)
ENTRY_POSE_TOL     = 0.2              # live policy-entry pose gate: max leg |q-q_default| <= 0.2 rad (r2)

# ── r15 factory stand-up replay (recorded with --record; ledger §5.10: ramp stand-up collapsed the rear) ──
STANDUP_TRAJ_DEFAULT  = "logs/factory_traj_standup.npz"  # default location when run from the package root
REPLAY_BLEND_IN_S     = 1.2           # current pose -> first trajectory frame (prone); refold first even when half-collapsed
REPLAY_BLEND_OUT_S    = 0.5           # last trajectory frame -> q_default (recording ends <=0.07 rad from it)
REPLAY_TRIM_PREROLL_S = 0.5           # cut the still head of the recording, keep this much pre-roll before motion
REPLAY_END_TOL        = 0.3           # final frame farther than this from q_default = not a stand-up recording, reject
NEAR_STAND_TOL        = 0.35          # pressing 1 while already near the stance -> skip the replay, ramp-trim directly
REPLAY_WHEEL_KD       = 0.5           # soft wheel damping during the replay (wheels roll 4~6 rad passively; braking fights the tuck)
TAU_IMPACT            = 40.0          # instant impact-stop threshold (motor peak 48, driver self-limits; r17's 32 was grazed by foot strikes)
TAU_OVERLOAD          = 32.0          # sustained-overload threshold: one joint |tau| above this for OVERLOAD_TICKS -> stop (motor-protection philosophy)
OVERLOAD_TICKS        = 30            # = 600 ms. Walking p90 20.3 and transition peaks ~30 last 1~3 ticks; standing front
                                      # hips measured ~29 (<32, no trip). Only a true jam/stall pins one joint >32 for 0.6 s
OVERLOAD_WARN         = 15            # print one warning at 300 ms (no e-stop yet)
CMD_SLEW_XY           = 2.0           # r19 command slew: vx/vy limited to 2 m/s² (1 m/s brakes in 0.5 s; 0=off)
CMD_SLEW_WZ           = 4.0           # wz limited to 4 rad/s² (scales with --cmd_slew)

# ⚠ keep +1 until gate #10 is done; stage1 (brake, v=0) is sign-independent, stage2 REQUIRES measured signs (see --wheel_test)
WHEEL_SIGN = (1.0, 1.0, 1.0, 1.0)     # raw -> urdf wheel-speed sign, SDK leg order FR, FL, BR, BL

# ── raw<->urdf map (validated on-robot by v1 aav_deploy.py; tabulated here over the SDK flat index k=leg*4+j) ──
#    urdf = S*raw + C ;  raw = S*(urdf - C) ;  velocities: urdf_dot = S*raw_dot
S16 = np.zeros(16, np.float32)
C16 = np.zeros(16, np.float32)
for _leg in range(4):
    _right, _front = _leg in (0, 2), _leg in (0, 1)        # lowlevel.h: FR=0 FL=1 BR=2 BL=3
    S16[_leg * 4 + 0], C16[_leg * 4 + 0] = (-1.0 if _front else 1.0), (0.4887 if _right else -0.4887)
    S16[_leg * 4 + 1], C16[_leg * 4 + 1] = (1.0 if _right else -1.0), 2.967
    S16[_leg * 4 + 2], C16[_leg * 4 + 2] = (1.0 if _right else -1.0), -2.723
    S16[_leg * 4 + 3], C16[_leg * 4 + 3] = WHEEL_SIGN[_leg], 0.0

def raw_to_urdf16(r):  return S16 * np.asarray(r, np.float32) + C16
def urdf_to_raw16(u):  return S16 * (np.asarray(u, np.float32) - C16)

# leg-joint URDF limits, policy order (abad,hip,knee)x4, 1e-3 margin (v1 values, same asset)
LEG_LO = np.array([-0.4887, -1.152, -2.723] * 4, np.float32) + 1e-3
LEG_HI = np.array([0.4887, 2.967, -0.602] * 4, np.float32) - 1e-3


def load_standup_traj(path, speed=1.0):
    """Load the factory stand-up recorded with --record -> (t (N,), legs12_urdf (N,12) in policy order).
    raw->urdf uses the same map as dispatch; the still head is cut (keeping REPLAY_TRIM_PREROLL_S of
    pre-roll); leg targets are clamped to the limits; the final frame must sit near q_default (otherwise it
    is not a stand-up recording). Any failure returns None (caller falls back to the straight ramp).
    speed: playback-rate multiplier, 1 = original speed, 0.8 = slowed to 80%."""
    if not path or not os.path.exists(path):
        print("  [standup] trajectory not found: %s" % path)
        return None
    try:
        d = np.load(path)
        t = np.asarray(d["t"], np.float64)
        raw = np.asarray(d["raw"], np.float32)
    except Exception as e:
        print("  [standup] failed to read trajectory: %s" % e)
        return None
    if len(t) < 50 or raw.ndim != 2 or raw.shape[1] != 16 or len(raw) != len(t):
        print("  [standup] bad trajectory shape: t%s raw%s" % (t.shape, raw.shape))
        return None
    legs = np.stack([np.clip(np.asarray(sdk_to_policy(raw_to_urdf16(raw[i]).tolist())[0],
                                        np.float32), LEG_LO, LEG_HI) for i in range(len(t))])
    end_err = float(np.abs(legs[-1] - Q_DEFAULT_LEGS).max())
    if end_err > REPLAY_END_TOL:
        print("  [standup] final frame %.2f rad from q_default > %.1f — not a stand-up recording? rejected" % (end_err, REPLAY_END_TOL))
        return None
    mov = np.where(np.abs(legs - legs[0]).max(axis=1) > 0.05)[0]       # cut the still head
    i0 = int(np.searchsorted(t, max(t[mov[0]] - REPLAY_TRIM_PREROLL_S, t[0]))) if len(mov) else 0
    sp = min(max(float(speed), 0.3), 1.5)
    if sp != float(speed):
        print("  [standup] traj_speed %.2f outside 0.3~1.5, clamped to %.2f" % (speed, sp))
    if sp > 1.2:
        print("  [standup] ⚠ replaying faster than the factory (%.2fx) — lag and transient torques grow" % sp)
    t2 = ((t[i0:] - t[i0]) / sp).astype(np.float32)
    return t2, legs[i0:]

# ── shared-memory layout (lowlevel.h; validated on-robot by v1) ──────────────────────────────────────────
#   cmd  @0   : legs[4] x (abad,hip,knee,foot each 5xf32 [p,v,kp,kd,tau] + i32 flags) = 4x84 = 336
#   cf   @336 : consumer_flags[2], write (1,0) to assert CONSUMER_CONTROL
#   state@344 : legs[4] x (4 x (i32 flags, f32 p, f32 v, f32 tau)) = 4x64 = 256
CMD_OFF, CMD_CF_OFF, STATE_OFF, SHM_SIZE = 0, 336, 344, 10240


class Shm:
    """spline_shm / imu_shm access. state is read-only; cmd is built as one block and written at once.
    Nothing is written in dry-run."""

    def __init__(self, live):
        self.live = live
        fd = os.open("/dev/shm/spline_shm", os.O_RDWR if live else os.O_RDONLY)
        self.sp = mmap.mmap(fd, SHM_SIZE,
                            prot=(mmap.PROT_READ | mmap.PROT_WRITE) if live else mmap.PROT_READ)
        os.close(fd)
        fd = os.open("/dev/shm/imu_shm", os.O_RDONLY)
        self.imu = mmap.mmap(fd, 1024, prot=mmap.PROT_READ)
        os.close(fd)
        self._q = np.zeros(16, np.float32)
        self._qd = np.zeros(16, np.float32)
        self._tau = np.zeros(16, np.float32)
        self._cmdbuf = bytearray(344)

    def read_joints(self):
        b = self.sp[STATE_OFF:STATE_OFF + 256]
        for leg in range(4):
            for j in range(4):
                _f, p, v, t = struct.unpack_from("<ifff", b, leg * 64 + j * 16)
                k = leg * 4 + j
                self._q[k], self._qd[k], self._tau[k] = p, v, t
        return self._q, self._qd, self._tau

    def read_imu(self):
        ts, ax, ay, az, gx, gy, gz, q0, q1, q2, q3 = struct.unpack_from("<Qffffffffff", self.imu, 0)
        # Quaternion measured to be xyzw (the wxyz in the lowlevel.h comment is wrong) — hand the raw
        # 4-tuple to the runtime, which interprets it as xyzw and enforces gate #12
        return (ts, np.array([gx, gy, gz], np.float32), (q0, q1, q2, q3),
                np.array([ax, ay, az], np.float32))

    def build_cmd(self, p16, v16, kp16, kd16):
        mv = memoryview(self._cmdbuf)
        for leg in range(4):
            for j in range(4):
                k = leg * 4 + j
                struct.pack_into("<fffff", mv, leg * 84 + j * 20,
                                 float(p16[k]), float(v16[k]), float(kp16[k]), float(kd16[k]), 0.0)
            struct.pack_into("<i", mv, leg * 84 + 80, 1)          # header comment: always set to 1
        struct.pack_into("<II", mv, CMD_CF_OFF, 1, 0)             # CONSUMER_CONTROL
        return self._cmdbuf

    def push_cmd(self, buf):
        if self.live:
            self.sp[0:344] = buf


class Keys:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        try:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.ok = True
        except Exception:
            self.ok = False

    def get(self):
        if not self.ok:
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None

    def restore(self):
        if self.ok:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def set_rt():
    try:
        os.sched_setaffinity(0, set(RT_CORES))
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(RT_PRIO))
        import resource
        resource.setrlimit(resource.RLIMIT_RTTIME, (200000, 200000))   # a runaway RT thread is killed after 200 ms
        return "SCHED_FIFO %d @ cores %s" % (RT_PRIO, RT_CORES)
    except PermissionError:
        try:
            os.sched_setaffinity(0, set(RT_CORES))
            return "affinity %s only (RT scheduling needs sudo)" % (RT_CORES,)
        except Exception:
            return "no RT, no core affinity"
    except Exception as e:
        return "RT setup failed: %s" % e


class Recorder:
    """Contract §9: 600 frames of [obs570|act16] -> dump.npy (directly usable by probe section E) + a rich
    npz; anomalies trigger a ring-buffer save.

    r2: disk writes run on a background thread (the 50 Hz control thread must not block on np.savez —
    the dry-run measured 31 ms peaks for a synchronous write); bursts have a cooldown (at least
    BURST_COOLDOWN_S apart, so a prone pose or violent motion cannot chain-trigger spam)."""

    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.dump, self.ring, self.trig, self.saved600 = [], [], 0, False
        self.last_burst_t = -1e9
        self._threads = []

    def policy_step(self, obs, act, extras, anomalous):
        row = np.concatenate([obs, act]).astype(np.float32)
        if not self.saved600:
            self.dump.append((row, extras))
            if len(self.dump) >= DUMP_FRAMES:
                self._save(self.dump, "dump")
                self.saved600 = True
                self.dump = []
        self.ring.append((row, extras))
        if len(self.ring) > RING_FRAMES:
            self.ring.pop(0)
        if anomalous and self.trig == 0 and (time.monotonic() - self.last_burst_t) > BURST_COOLDOWN_S:
            self.trig = RING_FRAMES // 2                    # record half a window more: window = half before + half after
        elif self.trig > 0:
            self.trig -= 1
            if self.trig == 0:
                self.last_burst_t = time.monotonic()
                self._save(self.ring, "burst")

    def save_now(self, tag="manual", note=None):
        if self.ring:
            self._save(self.ring, tag, note)
        elif note is not None:                              # keep the trigger snapshot even with an empty ring (never entered policy)
            self._save_note_only(tag, note)

    def _save_note_only(self, tag, note):
        import json
        fp = os.path.join(self.outdir, "%s_%s.json" % (tag, time.strftime("%Y%m%d_%H%M%S")))
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(note, f, ensure_ascii=False, indent=1)
        print("\n[rec] trigger snapshot -> %s (ring buffer empty)" % fp)

    def flush(self, timeout=10.0):
        """Wait for background writes to finish (call before exiting / before reading the files)."""
        for t in self._threads:
            t.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    def _save(self, rows, tag, note=None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        mat = np.stack([r[0] for r in rows])                # array assembly stays on the control thread (pure memcpy, ~ms)
        extras = {k: np.stack([r[1][k] for r in rows]) for k in rows[0][1]}
        n = len(rows)

        def _write():
            if tag == "dump":                               # contract §9 fixed name; probe E reads columns 0:570/570:586
                np.save(os.path.join(self.outdir, "dump.npy"), mat)
            np.savez(os.path.join(self.outdir, "%s_%s.npz" % (tag, stamp)), obs_act=mat, **extras)
            if note is not None:                            # r17: trigger-frame snapshot (the ring does not hold the trigger tick)
                import json
                with open(os.path.join(self.outdir, "%s_%s.json" % (tag, stamp)), "w", encoding="utf-8") as f:
                    json.dump(note, f, ensure_ascii=False, indent=1)
            print("\n[rec] saved %d frames -> %s/%s_%s.npz%s" % (
                n, self.outdir, tag, stamp, " (+dump.npy)" if tag == "dump" else
                (" (+trigger snapshot .json)" if note is not None else "")))

        th = threading.Thread(target=_write, daemon=True)   # background write (r2)
        th.start()
        self._threads.append(th)


def wheel_profile(profile, a=None):
    """(v_des_urdf4, kd) for the profile. stage1 = deploy-side brake, policy wheel outputs discarded (contract §4.1)."""
    if profile == "stage1":
        return np.zeros(4, np.float32), WHEEL_KD_BRAKE
    v = np.zeros(4, np.float32) if a is None else (WHEEL_SCALE * np.asarray(a[12:16], np.float32))
    return v, WHEEL_KD_ROLL


def _robot_launch_bin():
    return "/usr/local/bin/robot-launch" if os.path.exists("/usr/local/bin/robot-launch") else "robot-launch"


def factory_stop(egg="motion_control", timeout=6.0):
    """Stop the factory egg (default motion_control) and poll until mc_ctrl is really gone. Only a True
    return makes the takeover safe. It is a separate egg from spline_daemon, so stopping it leaves the
    encoder refresh running (ledger §5.9)."""
    import subprocess
    try:
        r = subprocess.run([_robot_launch_bin(), "stop", egg], capture_output=True, text=True, timeout=timeout)
        msg = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        print("  [factory] stop %s: %s" % (egg, msg[0] if msg else "ok"))
    except FileNotFoundError:
        print("  [factory] robot-launch not found — skipping (stop mc_ctrl manually, then rerun with --no_factory_mgmt)")
        return False
    except Exception as e:
        print("  [factory] stop raised: %s" % e)
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            g = subprocess.run(["pgrep", "-x", "mc_ctrl"], capture_output=True, text=True, timeout=2.0)
        except Exception:
            break
        if g.returncode != 0:                       # pgrep found nothing = mc_ctrl is down
            print("  [factory] mc_ctrl stopped (%.1fs) — this tool is now the only writer" % (time.monotonic() - t0))
            return True
        time.sleep(0.2)
    print("  [factory] ⚠ waited %.0fs and mc_ctrl is still up — unsafe, aborting the takeover" % timeout)
    return False


def factory_start(egg="motion_control", timeout=6.0):
    """Restore the factory egg (called on exit)."""
    import subprocess
    try:
        r = subprocess.run([_robot_launch_bin(), "start", egg], capture_output=True, text=True, timeout=timeout)
        msg = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        print("  [factory] start %s: %s" % (egg, msg[0] if msg else "ok"))
        return True
    except Exception as e:
        print("  [factory] start raised (run %s start %s manually): %s" % (_robot_launch_bin(), egg, e))
        return False


# ═════════════════════════ --check: on-robot software gates (no shared-memory access) ═════════════════════════
def check_mode(policy_path, profile, standup_traj=STANDUP_TRAJ_DEFAULT):
    bad = 0

    def ck(name, ok, detail=""):
        nonlocal bad
        print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
        bad += 0 if ok else 1

    # C1 raw<->urdf round trip
    rng = np.random.RandomState(7)
    u = rng.uniform(-1, 1, 16).astype(np.float32)
    ck("C1 raw<->urdf random round trip", float(np.abs(raw_to_urdf16(urdf_to_raw16(u)) - u).max()) < 1e-5)
    qd_sdk = np.asarray(policy_to_sdk(list(Q_DEFAULT_LEGS), [0.0] * 4), np.float32)
    ck("C1b q_default round trip", float(np.abs(raw_to_urdf16(urdf_to_raw16(qd_sdk)) - qd_sdk).max()) < 1e-5)

    # C2 200 ideal standing steps (real policy; the zero stub without --policy is for 3090-side syntax
    # shakedown only — on the robot a real .onnx is mandatory)
    if policy_path:
        rt = LocoRuntimeV2(policy_path, profile)
    else:
        rt = LocoRuntimeV2.__new__(LocoRuntimeV2)
        rt.profile, rt.policy = profile, (lambda o: np.zeros(N_ACT, np.float32))
        from loco_runtime_v2 import HistoryBuffer
        rt.history, rt.a_prev, rt._imu_check_pending = HistoryBuffer(), np.zeros(N_ACT, np.float32), True
        print("[warn] C2 running with a zero-stub policy (no --policy) — use the real .onnx on the robot")
    rt.reset()
    level_q, level_acc = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 9.81)
    out = None
    for _ in range(200):
        out = rt.step([0, 0, 0], level_q, qd_sdk, [0.0] * 16, [0, 0, 0], acc=level_acc)
    a = np.asarray(out["action"], np.float32)
    legs_p, _ = sdk_to_policy(out["p_des"])
    abad_t = [abs(legs_p[i] - Q_DEFAULT_LEGS[i]) for i in (0, 3, 6, 9)]
    knee_t = [legs_p[i] for i in (2, 5, 8, 11)]
    ck("C2a finite outputs over 200 steps", bool(np.isfinite(a).all()))
    ck("C2b ABAD targets <= 0.4398", max(abad_t) <= 0.4398, "max %.4f" % max(abad_t))
    ck("C2c knee targets within hard limits [-2.723,-0.602]", all(-2.723 <= k <= -0.602 for k in knee_t),
       " ".join("%.3f" % k for k in knee_t))

    # C3 wheel dispatch per profile
    v4, kd_w = wheel_profile(profile, a)
    if profile == "stage1":
        ck("C3 stage1 wheels: v=0, kd=5.0 (brake, policy wheel outputs discarded)",
           float(np.abs(v4).max()) == 0.0 and kd_w == WHEEL_KD_BRAKE)
    else:
        ck("C3 stage2 wheels: v=5*a[12:16], kd=0.7",
           float(np.abs(v4 - WHEEL_SCALE * a[12:16]).max()) < 1e-6 and kd_w == WHEEL_KD_ROLL)

    # C4 leg dispatch: contract kp/kd + limit clamping
    kp_legs = [out["kp"][k] for k in range(16) if k not in WHEEL_SDK_INDICES]
    kd_legs = [out["kd"][k] for k in range(16) if k not in WHEEL_SDK_INDICES]
    ck("C4a legs kp=20 kd=0.7", all(x == LEG_KP for x in kp_legs) and all(x == LEG_KD for x in kd_legs))
    wild = Q_DEFAULT_LEGS + np.float32(10.0)
    ck("C4b limit clamp", bool((np.clip(wild, LEG_LO, LEG_HI) <= LEG_HI).all()
                            and (np.clip(wild, LEG_LO, LEG_HI) >= LEG_LO).all()))

    # C5 cmd buffer field-by-field round trip (no real shared memory; verifies the packed layout only)
    shm = Shm.__new__(Shm)
    shm.live, shm._cmdbuf = False, bytearray(344)
    p = rng.uniform(-2, 2, 16).astype(np.float32)
    v = rng.uniform(-3, 3, 16).astype(np.float32)
    kp = rng.uniform(0, 60, 16).astype(np.float32)
    kd = rng.uniform(0, 5, 16).astype(np.float32)
    buf = shm.build_cmd(p, v, kp, kd)
    ok = len(buf) == 344
    for leg in range(4):
        for j in range(4):
            k = leg * 4 + j
            pp, vv, pkp, pkd, tff = struct.unpack_from("<fffff", buf, leg * 84 + j * 20)
            ok &= (abs(pp - p[k]) < 1e-6 and abs(vv - v[k]) < 1e-6 and abs(pkp - kp[k]) < 1e-6
                   and abs(pkd - kd[k]) < 1e-6 and tff == 0.0)
        ok &= struct.unpack_from("<i", buf, leg * 84 + 80)[0] == 1
    ok &= struct.unpack_from("<II", buf, CMD_CF_OFF) == (1, 0)
    ck("C5 344B cmd layout field round trip + flags/CF", ok)

    # C6 STAND interpolation: from the worst pose (urdf=C, i.e. raw=0), <=0.05 rad per tick, endpoint = q_default
    q0_legs, _ = sdk_to_policy(raw_to_urdf16(np.zeros(16, np.float32)).tolist())
    q0_legs = np.asarray(q0_legs, np.float32)
    nsteps = int(STANDUP_SEC * POLICY_HZ)
    prev, max_step = q0_legs.copy(), 0.0
    for i in range(1, nsteps + 1):
        r = i / nsteps
        tgt = (1 - r) * q0_legs + r * Q_DEFAULT_LEGS
        max_step = max(max_step, float(np.abs(tgt - prev).max()))
        prev = tgt
    ck("C6 STAND <= %.2f rad per tick and endpoint=q_default" % STAND_STEP_MAX,
       max_step <= STAND_STEP_MAX and float(np.abs(prev - Q_DEFAULT_LEGS).max()) < 1e-5,
       "max step %.4f" % max_step)

    # C7 dump.npy shape and column ranges
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = Recorder(td)
        obs = np.zeros(OBS_DIM, np.float32)
        act = np.arange(N_ACT, dtype=np.float32)
        for _ in range(DUMP_FRAMES):
            rec.policy_step(obs, act, {"t": np.float32(0.0)}, False)
        rec.flush()
        d = np.load(os.path.join(td, "dump.npy"))
        ck("C7 dump.npy (%d,586), [570:586]=act" % DUMP_FRAMES,
           d.shape == (DUMP_FRAMES, 586) and bool((d[0, 570:586] == act).all()))

    # C8 stand-up replay trajectory (optional file: missing = skip with a notice about the ramp fallback, not a FAIL)
    st = load_standup_traj(standup_traj) if standup_traj else None
    if st is None:
        print("[skip] C8 stand-up trajectory unavailable (%s) — --live stand-up falls back to the straight ramp (poor rear-leg leverage; it has collapsed before)" % standup_traj)
    else:
        st_t, st_q = st
        dt2 = np.maximum(st_t[2:] - st_t[:-2], 1e-6)[:, None]
        vmax = float(np.abs((st_q[2:] - st_q[:-2]) / dt2).max())
        end_err = float(np.abs(st_q[-1] - Q_DEFAULT_LEGS).max())
        inlim = bool((st_q >= LEG_LO - 1e-6).all() and (st_q <= LEG_HI + 1e-6).all())
        ck("C8 stand-up replay trajectory usable", 5.0 <= float(st_t[-1]) <= 30.0 and end_err <= REPLAY_END_TOL
           and vmax < 6.0 and inlim,
           "duration %.1fs end err %.3f peak speed %.1f rad/s inside limits %s" % (st_t[-1], end_err, vmax, inlim))

    print("[check]", "ALL PASS — dry-run/--live may proceed" if bad == 0 else "%d FAILED — do not go on the robot" % bad)
    return 1 if bad else 0


# ═════════════════════════ live state machine ═════════════════════════
def run(args):
    runtime = LocoRuntimeV2(args.policy, args.profile)
    shm = Shm(args.live)
    rt_msg = set_rt()
    gc.disable()
    stand_kp = args.stand_kp if args.stand_kp > 0 else STAND_KP
    stand_kd = args.stand_kd if args.stand_kd > 0 else STAND_KD
    if stand_kp > 150:
        print("  [warn] stand_kp=%.0f is high; the stand-up may get stiff/oscillatory — raise it gradually" % stand_kp)
    tau_overload = args.tau_overload if args.tau_overload > 0 else TAU_OVERLOAD      # r18 motor-protection threshold
    tau_impact = max(args.tau_impact if args.tau_impact > 0 else TAU_IMPACT, tau_overload + 2.0)
    st_t = st_q = None                                      # r15: factory stand-up replay (optional)
    if not args.no_standup_replay:
        st = load_standup_traj(args.standup_traj, args.traj_speed)
        if st is not None:
            st_t, st_q = st
    if st_t is None and args.live:
        print("  [warn] no usable stand-up trajectory -> STAND falls back to the straight ramp (ledger §5.10: poor rear-leg leverage; ground stand-up has collapsed)")

    lock = threading.Lock()
    tgt = {"p": np.zeros(16, np.float32), "v": np.zeros(16, np.float32),
           "kp": np.zeros(16, np.float32), "kd": np.full(16, DAMP_KD, np.float32),
           "p_prev": np.zeros(16, np.float32), "smooth": False,   # r16: 500 Hz linear smoothing during STAND
           "stamp": time.monotonic()}
    stop = threading.Event()
    sender_n = [0]

    def sender():
        """Re-sends the last targets at 500 Hz (contract #11: keep refreshing even if the policy thread
        stalls; the watchdog only drops the gains, never the refresh)."""
        per = 1.0 / SEND_HZ
        nxt = time.monotonic()
        while not stop.is_set():
            with lock:
                age = time.monotonic() - tgt["stamp"]
                if tgt["smooth"]:
                    # r16: linear interpolation between two 50 Hz targets during STAND -> removes the
                    # torque ripple of the 50 Hz staircase (the clicking noise)
                    r = min(age * POLICY_HZ, 1.0)
                    p_use = tgt["p_prev"] + r * (tgt["p"] - tgt["p_prev"])
                else:
                    p_use = tgt["p"]
                buf = shm.build_cmd(p_use, tgt["v"], tgt["kp"], tgt["kd"])
                if age * 1000.0 > WATCHDOG_MS and float(tgt["kp"].max()) > 0.1:
                    print("\n⚠ watchdog: targets %.0f ms stale -> damping" % (age * 1000.0))
                    tgt["kp"][:] = 0.0
                    tgt["kd"][:] = DAMP_KD
                    tgt["v"][:] = 0.0
                    tgt["smooth"] = False
            shm.push_cmd(buf)
            sender_n[0] += 1
            nxt += per
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

    def set_targets(p_urdf_legs12, leg_kp, leg_kd, wheel_v_urdf4, wheel_kd, q_raw_now, smooth=False):
        """The single dispatch path (shared by STAND/HOLD/POLICY/jog): clamp -> urdf->raw -> targets to sender.
        smooth=True (STAND only): the sender interpolates between two ticks at 500 Hz; POLICY must stay
        False (training used zero-order hold)."""
        p12 = np.clip(np.asarray(p_urdf_legs12, np.float32), LEG_LO, LEG_HI)
        clipped = float(np.abs(p12 - np.asarray(p_urdf_legs12, np.float32)).max()) > 1e-6
        p_sdk = np.asarray(policy_to_sdk(p12.tolist(), [0.0] * 4), np.float32)
        p_raw = urdf_to_raw16(p_sdk)
        v_sdk = np.asarray(policy_to_sdk([0.0] * 12, list(wheel_v_urdf4)), np.float32)
        v_raw = S16 * v_sdk
        kp16 = np.full(16, leg_kp, np.float32)
        kd16 = np.full(16, leg_kd, np.float32)
        for i, w in enumerate(WHEEL_SDK_INDICES):
            p_raw[w] = q_raw_now[w]                     # kp=0 so position is meaningless; hold the reading as a guard
            kp16[w] = 0.0
            kd16[w] = wheel_kd
        with lock:
            tgt["p_prev"][:] = tgt["p"]
            tgt["p"][:], tgt["v"][:], tgt["kp"][:], tgt["kd"][:] = p_raw, v_raw, kp16, kd16
            tgt["smooth"] = bool(smooth)
            tgt["stamp"] = time.monotonic()
        return clipped

    def to_damping(why=""):
        q_raw = shm.read_joints()[0]
        with lock:
            tgt["p"][:] = q_raw
            tgt["p_prev"][:] = q_raw
            tgt["v"][:] = 0.0
            tgt["kp"][:] = 0.0
            tgt["kd"][:] = DAMP_KD
            tgt["smooth"] = False
            tgt["stamp"] = time.monotonic()
        if why:
            print("\n⚠ dropping to damping: %s" % why)
        return "damping"

    keys = Keys()
    rec = Recorder(args.dump_dir)
    th = None                      # the sender thread starts only once we are the sole writer (factory stopped)
    factory_managed = [False]

    def sig(_s, _f):
        stop.set()
    signal.signal(signal.SIGINT, sig)
    signal.signal(signal.SIGTERM, sig)
    signal.signal(signal.SIGHUP, sig)

    wv0, wkd0 = wheel_profile(args.profile)             # static wheel profile: STAND/HOLD also dispatch it, so the
    print("\n" + "=" * 62)                              # stage1 brake acts BEFORE policy entry — pushing a wheel by hand must feel viscous (contract §4.1)
    print("  mode     %s" % ("🔴 LIVE (writes cmd)" if args.live else "🟢 DRY-RUN (read-only)"))
    print("  sched    %s" % rt_msg)
    print("  policy   %s  [%s]" % (os.path.basename(args.policy), args.profile))
    print("  legs     kp %.0f / kd %.1f (stand-up %.0f/%.0f)   wheels %s" % (
        LEG_KP, LEG_KD, stand_kp, stand_kd,
        "v=0 kp=0 kd=5.0 brake" if args.profile == "stage1" else "v=5*a kd=0.7 (⚠ confirm gate #10 passed)"))
    print("  stand-up %s" % (
        "factory-replay %.1fs (blend-in %.1f / blend-out %.1f; %swheels soft-damped during replay kd%.1f, profile brake once standing)" % (
            st_t[-1], REPLAY_BLEND_IN_S, REPLAY_BLEND_OUT_S,
            "" if args.traj_speed == 1.0 else "%.2fx speed; " % args.traj_speed, REPLAY_WHEEL_KD)
        if st_t is not None else "straight ramp %.1fs (no trajectory)" % STANDUP_SEC))
    print("  torque e-stop  motor protection (r18): instant>%.0f=impact stop | one joint |tau|>%.0f for 600 ms=overload (normal walking/standing passes)"
          % (tau_impact, tau_overload))
    print("  cmd slew %s" % ("vx/vy %.1f m/s² wz %.1f rad/s² (1 m/s brakes in %.2fs; r19 anti nose-dive)" % (
        args.cmd_slew, args.cmd_slew * CMD_SLEW_WZ / CMD_SLEW_XY, 1.0 / args.cmd_slew)
        if args.cmd_slew > 0 else "OFF (step commands — hard braking at speed WILL pitch forward; debug only)"))
    print("=" * 62)
    if args.live:
        mgmt = not args.no_factory_mgmt
        print("⚠ LIVE. %sthe robot is hoisted and the e-stop is within reach. Type YES and Enter to continue: " % (
            "this tool will stop the factory mc_ctrl automatically and restore it on exit (x/Ctrl-C); confirm " if mgmt
            else "confirm mc_ctrl is stopped and "), end="", flush=True)
        if sys.stdin.readline().strip() != "YES":
            print("cancelled")
            keys.restore()
            return 0
        if mgmt:
            print("  stopping factory motion_control (it would fight us for spline_cmd — the gear-grinding root cause)…")
            if not factory_stop():
                print("  aborting startup. You may `robot-launch stop motion_control` manually and rerun with --no_factory_mgmt.")
                keys.restore()
                return 1
            factory_managed[0] = True
    th = threading.Thread(target=sender, daemon=True)      # sole writer confirmed; start the 500 Hz sender
    th.start()
    print("keys: 1=stand up  2=enter policy  SPACE=e-stop (damping)  x=exit & restore factory  c=zero cmd  r=save dump")
    print("      %s w/s fwd/back a/d strafe q/e turn%s\n" % (
        "hold" if not args.step_keys else "step",
        "  magnitudes %.2f/%.2f/%.2f, release<=0.6s zeros" % (HOLD_VX, HOLD_VY, HOLD_WZ) if not args.step_keys
        else "  step %.1f" % CMD_STEP))

    state = to_damping()
    cmd = np.zeros(3, np.float32)                           # operator target (written directly by keys)
    cmd_app = np.zeros(3, np.float32)                       # r19: applied command (slews toward the target)
    slew_step = (np.array([1.0, 1.0, CMD_SLEW_WZ / CMD_SLEW_XY], np.float32)
                 * args.cmd_slew / POLICY_HZ) if args.cmd_slew > 0 else None
    last_aL = 0.0
    stand_t0, stand_q0 = 0.0, np.zeros(12, np.float32)
    stand_mode = "ramp"                                     # r15: "replay" = factory stand-up replay
    overload_cnt = np.zeros(16, np.int32)                   # r18 per-joint sustained-overload counter (|tau|>tau_overload)

    def estop_note(why, st, tau_, grav_, cmd_):
        """Trigger-frame snapshot (by code order the ring buffer misses the trigger tick; the json keeps
        that tick's decisive data)."""
        return {"why": why, "state": st, "t": round(time.monotonic() - t_run0, 3),
                "tau16": [round(float(x), 2) for x in tau_],
                "qd16": [round(float(x), 2) for x in qd_urdf],
                "grav": [round(float(x), 3) for x in grav_],
                "cmd": [round(float(x), 2) for x in cmd_]}
    stats = {"ms": 0.0, "max_ms": 0.0, "n": 0, "late": 0, "clip": 0}
    last_print, last_key_t = time.monotonic(), 0.0
    last_ts, last_gyro, stale = -1, None, 0
    per = 1.0 / POLICY_HZ
    nxt = time.monotonic()
    t_run0 = time.monotonic()

    try:
        while not stop.is_set():
            t_a = time.perf_counter()
            q_raw, qd_raw, tau = shm.read_joints()
            ts, gyro, quat, acc = shm.read_imu()
            q_urdf = raw_to_urdf16(q_raw)
            qd_urdf = S16 * qd_raw
            try:
                grav = gravity_from_quat(quat)
            except ValueError:
                state = to_damping("invalid IMU quaternion")
                grav = np.array([0, 0, -1], np.float32)

            # IMU stream stall: ts AND gyro bit-identical for N consecutive ticks
            if ts == last_ts and last_gyro is not None and np.array_equal(gyro, last_gyro):
                stale += 1
            else:
                stale = 0
            last_ts, last_gyro = ts, gyro.copy()
            if stale >= IMU_STALE_TICKS and state in ("stand", "hold", "policy"):
                why = "suspected IMU stream stall (%d ticks unchanged)" % stale
                note = estop_note(why, state, tau, grav, cmd_app)
                state = to_damping(why)
                rec.save_now("estop", note=note)

            k = keys.get()
            if k:
                if k == " ":
                    state = to_damping("manual e-stop")
                elif k == "x":
                    state = to_damping("exit")
                    stop.set()                              # leave the main loop -> finally restores the factory stack
                elif k == "1" and state == "damping":
                    legs_now, _ = sdk_to_policy(q_urdf.tolist())
                    stand_q0[:] = legs_now
                    stand_t0 = time.monotonic()
                    near = float(np.abs(np.asarray(legs_now, np.float32)
                                        - Q_DEFAULT_LEGS).max()) <= NEAR_STAND_TOL
                    state = "stand"
                    if st_t is not None and not near:
                        stand_mode = "replay"
                        print("\n▶ standing up… (factory replay: blend-in %.1fs -> replay %.1fs -> blend-out %.1fs, kp %.0f, 500 Hz smoothed dispatch)"
                              "\n   the feet tuck under the body first and then push up (factory-identical motion, this is normal); wheels soft-damped during the replay, profile brake once standing"
                              % (REPLAY_BLEND_IN_S, st_t[-1], REPLAY_BLEND_OUT_S, stand_kp))
                    else:
                        stand_mode = "ramp"
                        print("\n▶ standing up… (%skp %.0f, %.1fs interpolation, dispatched through the action pipeline)" % (
                            "already near the stance, trimming directly: " if (near and st_t is not None) else "",
                            stand_kp, STANDUP_SEC))
                elif k == "2" and state == "hold":
                    legs_now = np.asarray(sdk_to_policy(q_urdf.tolist())[0], np.float32)
                    pose_err = float(np.abs(legs_now - Q_DEFAULT_LEGS).max())
                    if args.live and pose_err > ENTRY_POSE_TOL:
                        print("\n⚠ policy entry refused: max leg |q-q_default| = %.3f rad > %.2f —"
                              " STAND did not truly arrive (jammed leg / not enough gain?), staying in HOLD" % (pose_err, ENTRY_POSE_TOL))
                    else:
                        runtime.reset()                  # a_prev=0, history replicate-filled, gate #12 re-armed
                        cmd[:] = 0.0
                        cmd_app[:] = 0.0
                        state = "policy"
                        print("\n▶ policy takeover (pose err %.3f rad; no warm-up, no blending, closed loop at once; first step runs the IMU cross-check)"
                              % pose_err)
                elif k == "c":
                    cmd[:] = 0.0
                elif k == "r":
                    rec.save_now()
                elif not args.step_keys:
                    held = {"w": (HOLD_VX, 0, 0), "s": (-HOLD_VX, 0, 0),
                            "a": (0, HOLD_VY, 0), "d": (0, -HOLD_VY, 0),
                            "q": (0, 0, HOLD_WZ), "e": (0, 0, -HOLD_WZ)}.get(k)
                    if held is not None:
                        cmd[:] = held
                        last_key_t = time.monotonic()
                elif k == "w":
                    cmd[0] = min(cmd[0] + CMD_STEP, MAX_VX)
                elif k == "s":
                    cmd[0] = max(cmd[0] - CMD_STEP, -MAX_VX)
                elif k == "a":
                    cmd[1] = min(cmd[1] + CMD_STEP, MAX_VY)
                elif k == "d":
                    cmd[1] = max(cmd[1] - CMD_STEP, -MAX_VY)
                elif k == "q":
                    cmd[2] = min(cmd[2] + CMD_STEP, MAX_WZ)
                elif k == "e":
                    cmd[2] = max(cmd[2] - CMD_STEP, -MAX_WZ)
            if not args.step_keys and cmd.any() and (time.monotonic() - last_key_t) * 1000 > HOLD_RELEASE_MS:
                cmd[:] = 0.0
            # r19 command slew: the applied value slides toward the target under an acceleration limit —
            # releasing a key / reversing no longer steps straight to a hard brake (the direct trigger of the near nose-dive)
            if slew_step is not None:
                cmd_app += np.clip(cmd - cmd_app, -slew_step, slew_step)
            else:
                cmd_app[:] = cmd

            # automatic e-stops (contract §6; FALL_GRAV_Z=-0.5 ≈ 60°, stricter than |r/p|>1.6)
            # r18 turned the torque e-stop into a **motor protection** criterion (= the operator's concern:
            # "as long as the motors are fine"): the instant-hysteresis rule (r17) still false-tripped —
            # two e-stops proved that transition transients AND **zero-command standing** front hips graze
            # 28~30 (≈ kp20 x 1.4 rad target error), all normal operation near the 28 continuous rating.
            # The motor truly fears only two things: impact spikes and sustained-overload heating.
            # ① impact: max|tau| > tau_impact (default 40; peak rating 48, driver self-limits) -> stop now;
            # ② sustained overload: **one joint** with |tau| > tau_overload (default 32) for OVERLOAD_TICKS
            #   (600 ms) -> stop. Walking/transition >30 lasts 1~3 ticks (the running count resets);
            #   standing front hips ~29 < 32 never trip; only a jam/stall/abnormal pose pins one joint
            #   above 32 for 0.6 s. Whether to retrain for lower torque: ledger §5.14.
            if state in ("stand", "hold", "policy"):
                if grav[2] > FALL_GRAV_Z:
                    why = "tip-over projG_z=%+.2f" % grav[2]
                    note = estop_note(why, state, tau, grav, cmd_app)
                    state = to_damping(why)
                    rec.save_now("estop", note=note)
                elif args.live:
                    tmax = float(np.abs(tau).max())
                    overload_cnt = np.where(np.abs(tau) > tau_overload, overload_cnt + 1, 0)
                    jo = int(overload_cnt.argmax())
                    trip_why = ""
                    if tmax > tau_impact:
                        ji = int(np.abs(tau).argmax())
                        trip_why = "instant torque %.1f N*m (%s) > %.0f impact level" % (tmax, SDK_JOINT_NAMES[ji], tau_impact)
                    elif int(overload_cnt[jo]) >= OVERLOAD_TICKS:
                        trip_why = "joint %s sustained overload: |tau| %.1f N*m >%.0f for %dms (jam/stall/abnormal pose)" % (
                            SDK_JOINT_NAMES[jo], abs(float(tau[jo])), tau_overload, int(overload_cnt[jo]) * 20)
                    if trip_why:
                        note = estop_note(trip_why, state, tau, grav, cmd_app)
                        state = to_damping(trip_why)
                        rec.save_now("estop", note=note)
                        overload_cnt[:] = 0
                    elif int(overload_cnt[jo]) == OVERLOAD_WARN:
                        print("\n⚠ joint %s |tau| %.1f>%.0f for %dms (e-stop at %dms)" % (
                            SDK_JOINT_NAMES[jo], abs(float(tau[jo])), tau_overload,
                            OVERLOAD_WARN * 20, OVERLOAD_TICKS * 20))

            if state == "stand":
                t_e = time.monotonic() - stand_t0
                wheel_kd_now = wkd0
                if stand_mode == "replay":                   # r15: blend in -> replay the recording -> blend out
                    T = float(st_t[-1])
                    if t_e < REPLAY_BLEND_IN_S:
                        rr = t_e / REPLAY_BLEND_IN_S
                        tgt_legs = (1 - rr) * stand_q0 + rr * st_q[0]
                        wheel_kd_now = REPLAY_WHEEL_KD
                    elif t_e < REPLAY_BLEND_IN_S + T:
                        te = t_e - REPLAY_BLEND_IN_S
                        tgt_legs = np.array([np.interp(te, st_t, st_q[:, j])
                                             for j in range(12)], np.float32)
                        wheel_kd_now = REPLAY_WHEEL_KD
                    elif t_e < REPLAY_BLEND_IN_S + T + REPLAY_BLEND_OUT_S:
                        rr = (t_e - REPLAY_BLEND_IN_S - T) / REPLAY_BLEND_OUT_S
                        tgt_legs = (1 - rr) * st_q[-1] + rr * Q_DEFAULT_LEGS   # profile wheel brake returns from the settle phase
                    else:
                        tgt_legs = Q_DEFAULT_LEGS.copy()
                        state = "hold"
                else:
                    rr = min(t_e / STANDUP_SEC, 1.0)
                    tgt_legs = (1 - rr) * stand_q0 + rr * Q_DEFAULT_LEGS
                    if rr >= 1.0:
                        state = "hold"
                # contract §6: dispatch via the inverse of the policy action pipeline so scale/q_default/
                # reshape/PD/SDK all get exercised before the policy ever runs
                a_impl = (tgt_legs - Q_DEFAULT_LEGS) / ACTION_SCALE_LEGS
                p_cmd = Q_DEFAULT_LEGS + ACTION_SCALE_LEGS * a_impl
                set_targets(p_cmd, stand_kp, stand_kd, wv0, wheel_kd_now, q_raw, smooth=True)
                if state == "hold":
                    legs_meas = np.asarray(sdk_to_policy(q_urdf.tolist())[0], np.float32)
                    print("\n▶ standing complete (target=q_default, no sag compensation; measured max %.3f rad from q_default,"
                          " ~0.1-0.15 of gravity sag is normal). Push a wheel to confirm %s, then press 2 for policy" % (
                              float(np.abs(legs_meas - Q_DEFAULT_LEGS).max()),
                              "the viscous brake feel (stage1)" if args.profile == "stage1" else "light damping (stage2)"))

            elif state == "hold":
                set_targets(Q_DEFAULT_LEGS, stand_kp, stand_kd, wv0, wkd0, q_raw)

            elif state == "policy":
                try:
                    out = runtime.step(gyro, quat, q_urdf, qd_urdf, cmd_app, acc=acc)
                except RuntimeError as e:                 # gate #12 failure or any other refusal
                    state = to_damping("runtime refused: %s" % e)
                    continue
                a = np.asarray(out["action"], np.float32)
                last_aL = float(np.abs(a[:12]).max())
                if not np.isfinite(a).all():
                    state = to_damping("policy output NaN")
                    continue
                legs_p, _ = sdk_to_policy(out["p_des"])
                wv, wkd = wheel_profile(args.profile, a)
                clipped = set_targets(np.asarray(legs_p, np.float32), LEG_KP, LEG_KD, wv, wkd, q_raw)
                if clipped:
                    stats["clip"] += 1
                anomalous = float(np.abs(a).max()) > ANOM_A or float(np.abs(tau).max()) > ANOM_TAU
                rec.policy_step(out["obs"], a,
                                {"t": np.float32(time.monotonic() - t_run0), "cmd": cmd_app.copy(),
                                 "q_raw": q_raw.copy(), "qd_raw": qd_raw.copy(), "tau": tau.copy(),
                                 "gyro": gyro.copy(), "grav": grav.copy(),
                                 "quat": np.asarray(quat, np.float32)}, anomalous)

            ms = (time.perf_counter() - t_a) * 1000.0
            stats["ms"] += ms
            stats["n"] += 1
            stats["max_ms"] = max(stats["max_ms"], ms)
            if ms > per * 1000:
                stats["late"] += 1

            now = time.monotonic()
            if now - last_print > 0.5:
                dt_p = now - last_print if last_print else 1.0
                last_print = now
                hz = sender_n[0] / dt_p
                sender_n[0] = 0
                wq = qd_urdf[list(WHEEL_SDK_INDICES)]
                print("\r[%-7s] cmd(%+.1f,%+.1f,%+.1f) projGz%+.2f τmax%5.1f aL%4.1f "
                      "whl|qd|%4.1f %4.1f/%4.1fms late%d clip%d send%4.0fHz   " % (
                          state, cmd_app[0], cmd_app[1], cmd_app[2], grav[2], float(np.abs(tau).max()), last_aL,
                          float(np.abs(wq).max()), stats["ms"] / max(stats["n"], 1),
                          stats["max_ms"], stats["late"], stats["clip"], hz), end="", flush=True)
            if args.max_s and (now - t_run0) > args.max_s:
                stop.set()

            nxt += per
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()
    finally:
        to_damping("exit")
        time.sleep(0.3)
        stop.set()
        if th is not None:
            th.join(timeout=1.0)
        rec.flush()
        keys.restore()
        n = max(stats["n"], 1)
        print("\n\nexited (damping). mean %.2f ms, peak %.2f ms, late %d/%d, clipped %d" % (
            stats["ms"] / n, stats["max_ms"], stats["late"], n, stats["clip"]))
        if factory_managed[0]:
            print("restoring factory motion_control — the factory stack takes over (may stand/hold); keep the robot safe…")
            factory_start()
    return 0


# ═════════════════ --record: read-only high-rate recording of factory motion (for the stand-up replay) ═════════════════
def record_mode(args):
    """Read-only, high-rate (100 Hz) recording of joints + IMU. Use while the FACTORY locomotion drives
    the robot: within the recording window, use the factory remote to run [lie down -> stand up], capture
    the factory stand-up's full joint sequence, copy it back to the 3090 and inspect it with
    scripts/analyze_factory_traj.py, then build the replay stand-up from it."""
    shm = Shm(False)                                      # read-only; the factory keeps driving
    hz = 100.0
    dur = args.record_s if args.record_s > 0 else 10.0
    name = args.record
    os.makedirs(args.dump_dir, exist_ok=True)
    print("\nread-only recording %.0f Hz x %.0f s, name '%s'." % (hz, dur, name))
    print(">>> Drive the target motion with the factory remote NOW (e.g. lie down, then stand up). Recording started…")
    T, Q, QD, TAU, QUAT, ACC, GYRO = [], [], [], [], [], [], []
    per = 1.0 / hz
    t0 = time.monotonic()
    nxt = t0
    while time.monotonic() - t0 < dur:
        q, qd, tau = shm.read_joints()
        _ts, gyro, quat, acc = shm.read_imu()
        T.append(time.monotonic() - t0)
        Q.append(q.copy()); QD.append(qd.copy()); TAU.append(tau.copy())
        QUAT.append(np.asarray(quat, np.float32)); ACC.append(acc.copy()); GYRO.append(gyro.copy())
        if int((time.monotonic() - t0) * 2) % 2 == 0 and len(T) % 100 == 0:
            print("  …%.0f s" % (time.monotonic() - t0))
        nxt += per
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.monotonic()
    fp = os.path.join(args.dump_dir, "factory_traj_%s.npz" % name)
    np.savez(fp, t=np.asarray(T, np.float32), raw=np.stack(Q), qd=np.stack(QD), tau=np.stack(TAU),
             quat=np.stack(QUAT), acc=np.stack(ACC), gyro=np.stack(GYRO))
    print("saved %d frames -> %s" % (len(T), fp))
    print("copy back to the 3090: scp dog:.../logs/factory_traj_%s.npz outputs/  then analyze_factory_traj.py" % name)
    return 0


# ═════════════════ --snapshot: read-only capture of the current 16-joint pose (factory sign calibration) ═════════════════
def snapshot_mode(args):
    """Read-only capture of one frame: 16 raw joints + our urdf conversion + tau, saved as JSON. Use while
    the FACTORY locomotion drives: have the factory hold [lying] once (--snapshot lie) and [standing] once
    (--snapshot stand), copy both back to the 3090 and compare with scripts/analyze_pose_cal.py — that
    settles whether every joint's sign/zero agrees with our map."""
    import json
    shm = Shm(False)                                      # read-only; the factory keeps driving
    q_raw, qd_raw, tau = shm.read_joints()
    urdf = raw_to_urdf16(q_raw)
    name = args.snapshot
    os.makedirs(args.dump_dir, exist_ok=True)
    fp = os.path.join(args.dump_dir, "snapshot_%s.json" % name)
    data = {"name": name, "wall": time.time(),
            "raw16": [round(float(x), 5) for x in q_raw],
            "urdf16_ours": [round(float(x), 5) for x in urdf],
            "qd16": [round(float(x), 5) for x in qd_raw],
            "tau16": [round(float(x), 3) for x in tau]}
    with open(fp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print("\nread-only snapshot '%s' (factory holding this pose):" % name)
    print(" SDK            raw(encoder)  our-urdf    tau")
    for leg in range(4):
        for j in range(4):
            k = leg * 4 + j
            print("  legs[%d].%-4s  %+9.4f  %+9.4f  %+6.2f"
                  % (leg, ("abad", "hip", "knee", "foot")[j], q_raw[k], urdf[k], tau[k]))
    print("saved: %s" % fp)
    print("hint: capture 'lie' lying and 'stand' standing, copy both to the 3090; analyze_pose_cal.py does the comparison.")
    return 0


# ═════════════════ --probe: passive hand-swing verification (read-only, zero writes, robot hoisted) ═════════════════
def probe_mode(args):
    """Entirely read-only (spline_shm opened O_RDONLY — the kernel enforces that nothing is written), no
    cmd is ever dispatched — gear grinding / control fights are impossible.

    Preconditions: mc_ctrl stopped (joints limp, hand-movable), spline_daemon running (refreshing the
    encoders), robot hoisted. You swing each part by hand, one at a time; the program identifies which
    motor moved (SDK + policy index + naming), its direction and magnitude, prints a line per detection
    and records it; q / Ctrl-C prints the full 16-row table and saves it under logs/."""
    MOVE_THRESH, SETTLE_TICKS = 0.10, int(0.3 * POLICY_HZ)   # a peak must exceed MOVE_THRESH to count; 0.3 s of stillness ends one push event
    ENTER_THRESH, STILL_STEP = 0.05, 0.01                    # above ENTER = "being pushed"; per-tick travel < STILL_STEP = still
    DOMINANCE = 2.0          # the peak joint must move >= 2x the runner-up, otherwise "several moved at once" and nothing is recorded (anti cross-talk)
    KD_DAMP_LEG, KD_DAMP_WHEEL = 2.0, 0.5                # --damp: kp=0 pure damping; smooth to push, stays where released
    damp = bool(getattr(args, "damp", False))
    shm = Shm(True) if damp else Shm(False)              # --damp must write cmd (mc_ctrl stopped first); otherwise read-only
    damp_stop = threading.Event()
    if damp:
        _kd = np.full(16, KD_DAMP_LEG, np.float32)
        for _w in WHEEL_SDK_INDICES:
            _kd[_w] = KD_DAMP_WHEEL
        _dbuf = bytes(shm.build_cmd(np.zeros(16, np.float32), np.zeros(16, np.float32),
                                    np.zeros(16, np.float32), _kd))   # kp=0 throughout, p_des is meaningless

        def _damp_sender():
            per = 1.0 / SEND_HZ
            nxt = time.monotonic()
            while not damp_stop.is_set():
                shm.push_cmd(_dbuf)
                nxt += per
                d = nxt - time.monotonic()
                time.sleep(d) if d > 0 else None
                if d <= 0:
                    nxt = time.monotonic()
        threading.Thread(target=_damp_sender, daemon=True).start()
    keys = Keys()
    print_naming_table()
    if damp:
        print("\nActive damping + hand-swing check. This tool is writing pure damping kp=0 / kd=%.1f (wheels %.1f) — "
              "pushing should feel smooth, parts stay where released, no gear grinding. Preconditions: **mc_ctrl stopped** (this tool is the only writer), robot hoisted."
              % (KD_DAMP_LEG, KD_DAMP_WHEEL))
    else:
        print("\nPassive hand-swing check (read-only, no commands sent). Preconditions: robot already in damping (smooth to push, stays where released), hoisted.")
    print("  Swing one part at a time (suggested order: front-right leg abad->hip->knee->wheel, then front-left/back-right/back-left);")
    print("  push, then hold still ~0.3 s — the program grabs the PEAK of the push and reports one line (a springback also measures fine; it need not stay).")
    print("  For a wheel, roll it by hand toward the robot's FORWARD direction, then stop.")
    print("  keys: a/d previous/next motor (manual switch)   r print the table   z rebase   c clear and restart   q/Ctrl-C finish")
    print("  The prompt names which part to swing; a correct swing gets a ✔ but the pointer stays — press d yourself when ready for the next one.\n")

    base = shm.read_joints()[0].copy()
    q_prev = base.copy()
    table = {}            # sdk_idx -> dict(order, raw_d, urdf_d, peak)
    order = [0]
    in_event = False       # inside one "push" event?
    evt_peak_mag = 0.0     # peak travel of the dominant joint within the event
    evt_any_mag = 0.0      # peak travel of ANY joint within the event (cross-talk judgement)
    evt_joint = -1
    evt_peak_q = None      # 16-dim raw snapshot at the peak
    still_count = 0
    seen_motion = False
    t_last_live = time.monotonic()
    t_last_hint = time.monotonic()
    t_last_ambig = [0.0]
    sel = 0                                # policy index currently "expected to swing" (guidance pointer)
    import json
    os.makedirs(args.dump_dir, exist_ok=True)
    _ev_stamp = time.strftime("%Y%m%d_%H%M%S")
    _ev_path = os.path.join(args.dump_dir, "probe_events_%s.jsonl" % _ev_stamp)
    _ev_f = open(_ev_path, "w")            # standalone event log (one JSON line per detection, with base/peak snapshots)
    _t0 = time.monotonic()

    def _reset_event():
        nonlocal in_event, evt_peak_mag, evt_any_mag, evt_joint, evt_peak_q, still_count
        in_event = False
        evt_peak_mag = evt_any_mag = 0.0
        evt_joint = -1
        evt_peak_q = None
        still_count = 0

    def sdk_to_pi(sdk):
        leg, typ = sdk // 4, sdk % 4
        return (leg * 3 + typ) if typ < 3 else (12 + leg)

    def _announce():
        sdk, txt = describe_policy_idx(sel)
        st = "✓ measured" if sdk in table else "not yet"
        print("\n👉 now swing: [idx %d] %s   (%s; a previous / d next)" % (sel, txt.split("  [")[0], st))

    def _next_untested():
        for i in range(16):
            c = (sel + 1 + i) % 16
            if describe_policy_idx(c)[0] not in table:
                return c
        return None

    def record(sdk, peak_q, base_q):
        pi = sdk_to_pi(sdk)
        raw_d = float(peak_q[sdk] - base_q[sdk])
        urdf_d = float((raw_to_urdf16(peak_q) - raw_to_urdf16(base_q))[sdk])
        if sdk not in table:
            table[sdk] = {"order": order[0], "raw_d": raw_d, "urdf_d": urdf_d, "peak": abs(raw_d)}
            order[0] += 1
        elif abs(raw_d) > table[sdk]["peak"]:
            table[sdk].update(raw_d=raw_d, urdf_d=urdf_d, peak=abs(raw_d))
        _s, txt = describe_policy_idx(pi)
        part = txt.split("  [")[0]
        print("  ✔ detected: %s" % part)
        print("     SDK legs[%d].%s / policy idx %d | peak raw Δ %+.3f (%s) | urdf Δ %+.3f rad"
              % (sdk // 4, ("abad", "hip", "knee", "foot")[sdk % 4], pi, raw_d,
                 "increasing" if raw_d > 0 else "decreasing", urdf_d))
        return raw_d, urdf_d

    def final_table():
        rows = []
        print("\n════════════ joint mapping verification table (✔=detected while you swung, ✗=not measured) ════════════")
        print(" policy-idx  part (our naming)         SDK            seen  rawΔ   urdfΔ   peak-rad  swing#")
        for pi in range(16):
            sdk, txt = describe_policy_idx(pi)
            part = txt.split("  [")[0]
            t = table.get(sdk)
            mark = "✔" if t else "✗"
            if t:
                line = "  %2d    %-22s  legs[%d].%-4s   %s   %+5.2f  %+5.2f   %5.2f    %d" % (
                    pi, part, sdk // 4, ("abad","hip","knee","foot")[sdk % 4], mark,
                    t["raw_d"], t["urdf_d"], t["peak"], t["order"])
            else:
                line = "  %2d    %-22s  legs[%d].%-4s   %s      —      —       —      —" % (
                    pi, part, sdk // 4, ("abad","hip","knee","foot")[sdk % 4], mark)
            print(line)
            rows.append({"pi": pi, "sdk": sdk, "part": part, "seen": bool(t),
                         "raw_d": t["raw_d"] if t else 0.0, "urdf_d": t["urdf_d"] if t else 0.0,
                         "peak": t["peak"] if t else 0.0, "order": t["order"] if t else -1})
        missed = [r["pi"] for r in rows if not r["seen"]]
        print("══════════════════════════════════════════════════════════════════")
        if missed:
            print("policy indices not yet measured: %s — these parts have not been swung" % missed)
        else:
            print("all 16 motors measured ✔")
        try:
            os.makedirs(args.dump_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(args.dump_dir, "probe_table_%s.txt" % stamp)
            with open(fp, "w") as f:
                f.write("policy_idx\tpart\tSDK_leg\tSDK_type\tseen\traw_d\turdf_d\tpeak\tswing_order\n")
                for r in rows:
                    f.write("%d\t%s\tlegs[%d]\t%s\t%d\t%+.4f\t%+.4f\t%.4f\t%d\n" % (
                        r["pi"], r["part"], r["sdk"] // 4, ("abad","hip","knee","foot")[r["sdk"] % 4],
                        int(r["seen"]), r["raw_d"], r["urdf_d"], r["peak"], r["order"]))
            print("saved: %s" % fp)
            print("event log (per-detection with snapshots, for offline analysis): %s" % _ev_path)
        except Exception as e:
            print("(table printed; writing the file failed: %s)" % e)

    _announce()
    try:
        while True:
            q, _qd, _tau = shm.read_joints()
            now = time.monotonic()
            d = q - base
            ad = np.abs(d)
            srt = np.argsort(ad)[::-1]
            best = int(srt[0])
            bestmag = float(ad[best])
            second = float(ad[int(srt[1])])
            dominant = bestmag >= DOMINANCE * max(second, 1e-6)
            step = float(np.abs(q - q_prev).max())       # per-tick travel — stillness test (independent of the velocity field, more robust)
            q_prev = q.copy()

            if bestmag > ENTER_THRESH:                    # being pushed
                in_event = True
                seen_motion = True
                if bestmag > evt_any_mag:
                    evt_any_mag = bestmag
                if dominant and bestmag > evt_peak_mag:   # update the peak joint only while it dominates alone
                    evt_peak_mag, evt_joint, evt_peak_q = bestmag, best, q.copy()

            if step < STILL_STEP:
                still_count += 1
            else:
                still_count = 0
                t_last_live = now

            if in_event and still_count >= SETTLE_TICKS:  # one push finished (came to rest — stayed or sprang back, both fine)
                if evt_peak_mag > MOVE_THRESH and evt_joint >= 0 and evt_peak_q is not None:
                    raw_d, urdf_d = record(evt_joint, evt_peak_q, base)
                    actual_pi = sdk_to_pi(evt_joint)
                    match = (actual_pi == sel)
                    if match:
                        print("     ✔ matches the current expectation [idx %d]" % sel)
                    else:
                        _es, _et = describe_policy_idx(sel)
                        print("     ⚠ expected [idx %d] %s but [idx %d] moved — you may have swung another part, or the mapping disagrees"
                              % (sel, _et.split("  [")[0], actual_pi))
                    try:
                        _ev_f.write(json.dumps({
                            "t": round(now - _t0, 3), "expected_pi": sel, "actual_pi": actual_pi,
                            "actual_sdk": evt_joint, "match": match, "raw_d": round(raw_d, 5),
                            "urdf_d": round(urdf_d, 5), "peak": round(float(evt_peak_mag), 5),
                            "base_q": [round(float(x), 5) for x in base],
                            "peak_q": [round(float(x), 5) for x in evt_peak_q]}, ensure_ascii=False) + "\n")
                        _ev_f.flush()
                    except Exception:
                        pass
                    base = q.copy()                       # new baseline: stayed -> here; sprang back -> ≈ the old one
                    q_prev = q.copy()
                    # no auto-advance: the pointer moves only on a/d (a correct swing gets its ✔; press d for the next)
                elif evt_any_mag > MOVE_THRESH and now - t_last_ambig[0] > 2.0:
                    print("  ⚠ several joints moved together with no single dominant one — swing ONE part at a time; press z to rebase and retry if needed")
                    t_last_ambig[0] = now
                _reset_event()

            k = keys.get()
            if k in ("q", "\x03"):
                break
            elif k == "a":
                sel = (sel - 1) % 16
                _announce()
            elif k == "d":
                sel = (sel + 1) % 16
                _announce()
            elif k == "r":
                final_table()
            elif k == "z":
                base = q.copy()
                q_prev = q.copy()
                _reset_event()
                print("  [baseline reset]")
            elif k == "c":
                table.clear()
                order[0] = 0
                base = q.copy()
                q_prev = q.copy()
                _reset_event()
                print("  [cleared, starting over]")
            if now - t_last_hint > 10.0 and not seen_motion and now - t_last_live > 10.0:
                print("  (no clear joint motion for 10 s — confirm the robot is hoisted, joints hand-movable, spline_daemon running;"
                      " push then stop is enough, a springback also measures)")
                t_last_hint = now
            time.sleep(1.0 / POLICY_HZ)
    finally:
        damp_stop.set()
        time.sleep(0.05)
        keys.restore()
        final_table()
        try:
            _ev_f.close()
        except Exception:
            pass
    return 0


# ═════════════════ joint naming (fact sources: simulation asset + lowlevel.h, see joint_map_v2.py) ═════════════════
# SDK leg order legs[0..3] = front-right FR / front-left FL / back-right BR / back-left BL; the 4 motors in
# each leg are, in order: abad, hip, knee, foot (wheel)
# Policy (action/observation) order: legs grouped FR,FL,BR,BL with abad,hip,knee each (12 total), then the
# 4 wheels FR,FL,BR,BL
_SDK_LEG_DESC = ("front-right leg FR", "front-left leg FL", "back-right leg BR", "back-left leg BL")
_TYPE_DESC = ("abad — whole-leg lateral ab/adduction about the body",
            "hip  — thigh swinging fore/aft about the shoulder",
            "knee — shank extend/flex",
            "foot — the wheel at the foot, rolling")
_Q_DEF_STR = ("0.00", "0.80", "-1.50")   # q_default for abad/hip/knee (wheel position is meaningless)


def describe_policy_idx(pi):
    """Policy index pi (0-11 legs, 12-15 wheels) -> (sdk_idx, human-readable description)."""
    if pi < 12:
        leg, typ = pi // 3, pi % 3
        sdk = leg * 4 + typ
        txt = ("%s · %s  [policy idx %d / SDK legs[%d].%s / q_default %s rad]"
               % (_SDK_LEG_DESC[leg], _TYPE_DESC[typ], pi, leg, ("abad", "hip", "knee")[typ], _Q_DEF_STR[typ]))
        return sdk, txt
    k = pi - 12
    sdk = k * 4 + 3
    return sdk, "%s · %s  [policy idx %d / SDK legs[%d].foot]" % (_SDK_LEG_DESC[k], _TYPE_DESC[3], pi, k)


def print_naming_table():
    print("\njoint naming reference (I describe parts in these terms; verify against the physical robot) ——————")
    print("  SDK leg order: legs[0]=front-right FR  legs[1]=front-left FL  legs[2]=back-right BR  legs[3]=back-left BL")
    print("  4 motors per leg, in order: [0]abad (lateral) [1]hip (thigh) [2]knee (shank) [3]foot (wheel)")
    print("  full 16-motor list (selection order = policy index):")
    for pi in range(16):
        _sdk, txt = describe_policy_idx(pi)
        print("   %2d) %s" % (pi, txt))
    print("————————————————————————————————————————————————————————————————————")


# ═════════════════ --inspect: interactive per-motor verification (robot MUST be hoisted) ═════════════════
def inspect_mode(args):
    """a/d selects previous/next motor and prints the selection; SPACE runs a small confirmation move;
    r reprints; q/Ctrl-C exits.

    Leg joints: a +0.15 rad "out-hold-back" triangle from q_default (URDF positive direction), reading
    back travel and cross-talk. Wheels: a +2 rad/s pulse for 1.2 s. Every move goes through EXACTLY the
    same dispatch pipeline as --live (policy -> SDK -> raw -> motor)."""
    DELTA_LEG, WHEEL_PULSE, RAMP_S, HOLD_S, PULSE_S = 0.15, 2.0, 0.6, 0.6, 1.2
    shm = Shm(args.live)
    set_rt()
    lock = threading.Lock()
    tgt = {"p": np.zeros(16, np.float32), "v": np.zeros(16, np.float32),
           "kp": np.zeros(16, np.float32), "kd": np.full(16, DAMP_KD, np.float32)}
    stop = threading.Event()

    def sender():
        per = 1.0 / SEND_HZ
        nxt = time.monotonic()
        while not stop.is_set():
            with lock:
                buf = shm.build_cmd(tgt["p"], tgt["v"], tgt["kp"], tgt["kd"])
            shm.push_cmd(buf)
            nxt += per
            d = nxt - time.monotonic()
            time.sleep(d) if d > 0 else None
            if d <= 0:
                nxt = time.monotonic()

    def set_hold(legs12_urdf, sel_wheel=None, wheel_v=0.0):
        """Hold all legs at legs12_urdf (kp=STAND), wheels braked; with sel_wheel given, that wheel gets a
        wheel_v velocity command instead."""
        q_raw_now = shm.read_joints()[0]
        p12 = np.clip(np.asarray(legs12_urdf, np.float32), LEG_LO, LEG_HI)
        p_raw = urdf_to_raw16(np.asarray(policy_to_sdk(p12.tolist(), [0.0] * 4), np.float32))
        kp16 = np.full(16, STAND_KP, np.float32)
        kd16 = np.full(16, STAND_KD, np.float32)
        v16 = np.zeros(16, np.float32)
        for i, w in enumerate(WHEEL_SDK_INDICES):
            p_raw[w], kp16[w], kd16[w] = q_raw_now[w], 0.0, WHEEL_KD_BRAKE
            if sel_wheel is not None and i == sel_wheel:
                v_sdk = np.zeros(16, np.float32)
                v_sdk[w] = wheel_v
                v16[w] = (S16 * v_sdk)[w]
                kd16[w] = WHEEL_KD_ROLL
        with lock:
            tgt["p"][:], tgt["v"][:], tgt["kp"][:], tgt["kd"][:] = p_raw, v16, kp16, kd16

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    print_naming_table()
    print("\n⚠ inspect mode (LIVE=%s). The robot MUST be hoisted, legs and wheels in the air, e-stop within reach." % args.live)
    print("  On entry it eases to the q_default stance over 2 s, then:")
    print("    a / d   select previous / next motor (prints the selection)")
    print("    SPACE   run the small confirmation move on the selected motor (legs ±0.15 rad out-and-back / wheels +2 rad/s pulse)")
    print("    r reprint selection   q or Ctrl-C exit (back to damping)")
    print("  Type YES and Enter to continue: ", end="", flush=True)
    if sys.stdin.readline().strip() != "YES":
        stop.set()
        return 0

    # ease to q_default over 2 s
    q0 = np.asarray(sdk_to_policy(raw_to_urdf16(shm.read_joints()[0]).tolist())[0], np.float32)
    n = int(STANDUP_SEC * POLICY_HZ)
    for i in range(n + 1):
        set_hold((1 - i / n) * q0 + (i / n) * Q_DEFAULT_LEGS)
        time.sleep(1.0 / POLICY_HZ)
    set_hold(Q_DEFAULT_LEGS)

    keys = Keys()
    sel = 0

    def announce():
        sdk, txt = describe_policy_idx(sel)
        print("\n▶ selected %s" % txt)
        if sel < 12:
            print("  SPACE: +%.2f rad out-and-back in the URDF positive direction. Verify that what moves is 「%s」 and note the direction (the signs are exactly what we are confirming)."
                  % (DELTA_LEG, txt.split("  [")[0]))
        else:
            print("  SPACE: +%.1f rad/s pulse for %.1fs. Verify that what spins is 「%s」 and which way the wheel turns under a positive command."
                  % (WHEEL_PULSE, PULSE_S, txt.split("  [")[0]))

    def do_leg(pi):
        base = Q_DEFAULT_LEGS.copy()
        goal = base.copy()
        goal[pi] = float(np.clip(base[pi] + DELTA_LEG, LEG_LO[pi], LEG_HI[pi]))
        steps = int(RAMP_S * POLICY_HZ)
        for ph, (a, b) in enumerate(((base, goal), (goal, goal), (goal, base))):
            hold = int(HOLD_S * POLICY_HZ) if ph == 1 else steps
            for i in range(hold + 1):
                r = i / max(hold, 1)
                cur = (1 - r) * a + r * b
                set_hold(cur)
                if ph == 1 and i == hold // 2:
                    q_now = np.asarray(sdk_to_policy(raw_to_urdf16(shm.read_joints()[0]).tolist())[0], np.float32)
                    tau = shm.read_joints()[2]
                    d_all = q_now - Q_DEFAULT_LEGS
                    sdk, _ = describe_policy_idx(pi)
                    print("  readback: selected joint Δ=%+.3f rad (commanded +%.3f)  tau=%.1f N*m" % (d_all[pi], DELTA_LEG, abs(tau[sdk])))
                    cross = [(j, d_all[j]) for j in range(12) if j != pi and abs(d_all[j]) > 0.05]
                    if cross:
                        print("  ⚠ cross-talk: " + "  ".join("idx%d(%+.3f)" % (j, d) for j, d in cross)
                              + "  — only the selected joint should move; cross-talk means weak holding stiffness or a mapping problem")
                time.sleep(1.0 / POLICY_HZ)
        set_hold(Q_DEFAULT_LEGS)
        print("  done, back at q_default.")

    def do_wheel(pi):
        k = pi - 12
        t0 = time.monotonic()
        qd_log = []
        while time.monotonic() - t0 < PULSE_S:
            set_hold(Q_DEFAULT_LEGS, sel_wheel=k, wheel_v=WHEEL_PULSE)
            sdk = k * 4 + 3
            qd_log.append(float(S16[sdk] * shm.read_joints()[1][sdk]))   # readback converted to the urdf direction
            time.sleep(1.0 / POLICY_HZ)
        set_hold(Q_DEFAULT_LEGS)
        mean_qd = float(np.mean(qd_log[len(qd_log) // 3:] or qd_log))
        print("  readback: that wheel's mean qd (urdf direction) %+.2f rad/s (commanded +%.1f). Note whether the wheel actually turned forward or backward." % (mean_qd, WHEEL_PULSE))
        print("  done.")

    try:
        announce()
        while not stop.is_set():
            # keep holding q_default while idle (the sender keeps refreshing the cmd)
            set_hold(Q_DEFAULT_LEGS)
            k = keys.get()
            if k is None:
                time.sleep(1.0 / POLICY_HZ)
                continue
            if k in ("q", "\x03"):
                break
            elif k == "a":
                sel = (sel - 1) % 16
                announce()
            elif k == "d":
                sel = (sel + 1) % 16
                announce()
            elif k == "r":
                announce()
            elif k == " ":
                if not args.live:
                    print("  [dry-run] no cmd written, logic only; add --live to actually move.")
                sdk, _ = describe_policy_idx(sel)
                print("  running…", flush=True)
                do_wheel(sel) if sel >= 12 else do_leg(sel)
    finally:
        q_raw = shm.read_joints()[0]
        with lock:
            tgt["p"][:], tgt["v"][:], tgt["kp"][:], tgt["kd"][:] = q_raw, 0.0, 0.0, DAMP_KD
        time.sleep(0.3)
        stop.set()
        th.join(timeout=1.0)
        keys.restore()
        print("\nback in damping, exiting.")
    return 0


# ═════════════════ on-robot helpers for gates #7 / #10 (robot MUST be hoisted) ═════════════════
def guided_mode(args, wheel):
    """--jog: per-joint jog to verify the mapping; --wheel_test: per-wheel raw +1 rad/s spin-direction mapping."""
    shm = Shm(args.live)
    set_rt()
    lock = threading.Lock()
    tgt = {"p": np.zeros(16, np.float32), "v": np.zeros(16, np.float32),
           "kp": np.zeros(16, np.float32), "kd": np.full(16, DAMP_KD, np.float32)}
    stop = threading.Event()

    def sender():
        per = 1.0 / SEND_HZ
        nxt = time.monotonic()
        while not stop.is_set():
            with lock:
                buf = shm.build_cmd(tgt["p"], tgt["v"], tgt["kp"], tgt["kd"])
            shm.push_cmd(buf)
            nxt += per
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

    def hold_legs(p_legs12, kp, kd, wheel_v_raw=None, wheel_kd=DAMP_KD):
        q_raw_now = shm.read_joints()[0]
        p12 = np.clip(np.asarray(p_legs12, np.float32), LEG_LO, LEG_HI)
        p_raw = urdf_to_raw16(np.asarray(policy_to_sdk(p12.tolist(), [0.0] * 4), np.float32))
        with lock:
            tgt["p"][:] = p_raw
            tgt["v"][:] = 0.0
            tgt["kp"][:] = kp
            tgt["kd"][:] = kd
            for i, w in enumerate(WHEEL_SDK_INDICES):
                tgt["p"][w], tgt["kp"][w], tgt["kd"][w] = q_raw_now[w], 0.0, wheel_kd
                if wheel_v_raw is not None:
                    tgt["v"][w] = wheel_v_raw[i]

    th = threading.Thread(target=sender, daemon=True)
    th.start()
    print("⚠ %s mode (LIVE=%s). The robot MUST be hoisted, legs and wheels in the air. Type YES and Enter to continue: " % (
        "wheel_test" if wheel else "jog", args.live), end="", flush=True)
    if sys.stdin.readline().strip() != "YES":
        stop.set()
        return 0
    try:
        # first interpolate to q_default over 2 s and hold
        q0_legs = np.asarray(sdk_to_policy(raw_to_urdf16(shm.read_joints()[0]).tolist())[0], np.float32)
        n = int(STANDUP_SEC * POLICY_HZ)
        for i in range(n + 1):
            hold_legs((1 - i / n) * q0_legs + (i / n) * Q_DEFAULT_LEGS, STAND_KP, STAND_KD)
            time.sleep(1.0 / POLICY_HZ)
        if wheel:
            print("\ngate #10: each wheel raw v=+1.0 rad/s for 2 s (legs held at q_default). Write down each wheel's actual spin direction!")
            for i, w in enumerate(WHEEL_SDK_INDICES):
                vr = np.zeros(4, np.float32)
                vr[i] = 1.0
                print("\n>> %s (SDK[%d]) raw v=+1.0 …" % (POLICY_WHEEL_NAMES[i], w), end="", flush=True)
                t0 = time.monotonic()
                qd_log = []
                while time.monotonic() - t0 < 2.0:
                    hold_legs(Q_DEFAULT_LEGS, STAND_KP, STAND_KD, wheel_v_raw=vr, wheel_kd=1.0)
                    qd_log.append(shm.read_joints()[1][w])
                    time.sleep(1.0 / POLICY_HZ)
                hold_legs(Q_DEFAULT_LEGS, STAND_KP, STAND_KD)
                print(" readback qd_raw mean %+.2f rad/s — note whether the wheel turned forward or backward" % float(np.mean(qd_log[25:] or qd_log)))
                time.sleep(0.5)
            print("\n-> After cross-checking against simulation (+v per wheel, drive direction), fill WHEEL_SIGN in this file; mandatory before stage2.")
        else:
            print("\ngate #7: each leg joint jogs +0.15 rad for 1.2 s. Verify that the moving physical joint matches its name!")
            for j in range(12):
                p = Q_DEFAULT_LEGS.copy()
                p[j] += 0.15
                sdk_name = SDK_JOINT_NAMES[(j // 3) * 4 + (j % 3)]
                print("\n>> policy[%2d] %s  (SDK %s)…" % (j, POLICY_LEG_JOINT_NAMES[j], sdk_name),
                      end="", flush=True)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1.2:
                    hold_legs(p, STAND_KP, STAND_KD)
                    time.sleep(1.0 / POLICY_HZ)
                q_now = np.asarray(sdk_to_policy(raw_to_urdf16(shm.read_joints()[0]).tolist())[0])
                print(" readback Δ=%+.3f rad (expect ≈+0.15)" % float(q_now[j] - Q_DEFAULT_LEGS[j]))
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.6:
                    hold_legs(Q_DEFAULT_LEGS, STAND_KP, STAND_KD)
                    time.sleep(1.0 / POLICY_HZ)
    finally:
        q_raw = shm.read_joints()[0]
        with lock:
            tgt["p"][:], tgt["v"][:], tgt["kp"][:], tgt["kd"][:] = q_raw, 0.0, 0.0, DAMP_KD
        time.sleep(0.3)
        stop.set()
        th.join(timeout=1.0)
        print("\nback in damping, exiting.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="", help="policy/*.onnx (use ONNX on the dog; .ts.pt needs torch)")
    ap.add_argument("--profile", default="stage1", choices=("stage1", "stage2"))
    ap.add_argument("--live", action="store_true", help="actually write cmd (mc_ctrl must be stopped first)")
    ap.add_argument("--check", action="store_true", help="on-robot software gates C1-C8, no shared-memory access")
    ap.add_argument("--jog", action="store_true", help="gate #7: per-joint jog (hoisted)")
    ap.add_argument("--wheel_test", action="store_true", help="gate #10: per-wheel raw spin-direction mapping (hoisted)")
    ap.add_argument("--inspect", action="store_true", help="interactive per-motor verification (hoisted, small active moves): a/d select, SPACE run")
    ap.add_argument("--probe", action="store_true", help="hand-swing verification (hoisted): swing by hand, auto-identified, table at the end")
    ap.add_argument("--snapshot", default="", metavar="NAME", help="read-only capture of one 16-joint pose to JSON (e.g. lie/stand, for factory sign calibration)")
    ap.add_argument("--record", default="", metavar="NAME", help="read-only high-rate joint+IMU recording (record the factory lie-down -> stand-up for the replay)")
    ap.add_argument("--record_s", type=float, default=0.0, help="recording duration in seconds (default 10)")
    ap.add_argument("--damp", action="store_true", help="with --probe: this tool writes kp=0/kd=2 pure damping (mc_ctrl must be stopped); smooth pushing, no gear grinding")
    ap.add_argument("--step_keys", action="store_true", help="stepped steering keys instead of hold-to-move")
    ap.add_argument("--dump_dir", default="logs", help="recording output directory (default ./logs)")
    ap.add_argument("--max_s", type=float, default=0.0, help="auto-exit after this many seconds (for shakedowns; 0=unlimited)")
    ap.add_argument("--force_profile", action="store_true", help="skip the policy-filename vs profile consistency check")
    ap.add_argument("--stand_kp", type=float, default=0.0, help="override the stand-up/HOLD leg kp (default 60; rear-heavy ground stand-ups may try 90/120)")
    ap.add_argument("--stand_kd", type=float, default=0.0, help="override the stand-up/HOLD leg kd (default 5)")
    ap.add_argument("--standup_traj", default=STANDUP_TRAJ_DEFAULT,
                    help="recorded stand-up trajectory npz for the replay (--record output; missing = fall back to the straight ramp)")
    ap.add_argument("--no_standup_replay", action="store_true", help="force the old straight-ramp stand-up (debugging)")
    ap.add_argument("--traj_speed", type=float, default=1.0,
                    help="stand-up replay speed multiplier (1=original; 0.8=slowed to 80%% speed; >1 faster, forbidden on a first live run)")
    ap.add_argument("--tau_overload", type=float, default=0.0,
                    help="sustained-overload threshold (default 32: one joint |tau|>this for 600 ms continuous trips the e-stop)")
    ap.add_argument("--tau_impact", type=float, default=0.0,
                    help="instant impact e-stop threshold (default 40; motor rating 28 continuous / 48 peak, driver self-limits)")
    ap.add_argument("--cmd_slew", type=float, default=CMD_SLEW_XY,
                    help="command slew: vx/vy acceleration limit in m/s² (wz scales 2x; default 2.0, a 1 m/s brake takes 0.5 s; 0=off)")
    ap.add_argument("--no_factory_mgmt", action="store_true", help="--live does not auto stop/start factory motion_control (stop mc_ctrl yourself)")
    args = ap.parse_args()

    if args.check:
        return check_mode(args.policy, args.profile, args.standup_traj)
    if args.snapshot:
        return snapshot_mode(args)
    if args.record:
        return record_mode(args)
    if args.probe:
        return probe_mode(args)
    if args.inspect:
        return inspect_mode(args)
    if not args.policy:
        print("[run] FAIL: --policy is required (use policy/*.onnx on the robot)")
        return 1
    if args.profile not in os.path.basename(args.policy) and not args.force_profile:
        print("[run] FAIL: policy filename %r does not contain %r — a stage1 ckpt with the stage2 profile = free wheels (the v1 accident)."
              " Add --force_profile only if you are certain" % (os.path.basename(args.policy), args.profile))
        return 1
    if args.wheel_test and args.profile == "stage2" and tuple(WHEEL_SIGN) == (1.0, 1.0, 1.0, 1.0):
        print("[warn] WHEEL_SIGN is still the +1 placeholder — this wheel_test run is what fills it in")
    if args.jog or args.wheel_test:
        return guided_mode(args, wheel=args.wheel_test)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
