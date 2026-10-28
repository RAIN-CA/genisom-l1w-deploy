# GENISOM L1-W v2 locomotion deployment (570-dim history observation, no odometry)

Deployment runtime for the GENISOM (zsibot) **L1-W wheeled-legged quadruped**, driving an RL policy
trained in Isaac Lab (task config `genisom_l1w_v2`). The actor is an MLP on a 10-frame proprioceptive
history (570 = 57 x 10) producing 16 joint commands (12 leg positions + 4 wheel velocities) at 50 Hz —
base velocity is inferred **implicitly** from the history; **there is no state estimator on the deploy
path**. The full I/O contract lives in `OBS_CONTRACT_V2.md` (shipped in the deploy package; read it first).

## Why a self-built low-level loop

The vendor SDK (`genisom_L1_sdk`) exposes only a **high-level** UDP/JSON interface (velocity / pose
commands plus telemetry). **No joint-level low-level API (per-joint position/velocity/kp/kd/torque) is
exposed for the wheeled variant** — but the RL policy needs exactly that, per joint at 50 Hz with an
explicit PD profile. This runtime therefore bypasses the SDK and talks to the motor stack directly
through the on-board shared memory:

- `/dev/shm/spline_shm` — motor command/state block consumed by the vendor's 1 ms spline daemon
  (layout read out of `lowlevel.h` and field-verified on hardware; see `run_loco_v2.py`);
- `/dev/shm/imu_shm` — IMU block (timestamp, acc, gyro, quaternion).

The factory controller `mc_ctrl` writes the same block, so a live session must be the **only writer**:
`--live` automatically runs `robot-launch stop motion_control` first (and restores it on exit). The two
things the vendor stack does not document had to be established experimentally and are baked in here:

1. **Joint mapping** — SDK leg order `legs[0..3] = FR, FL, BR, BL` with per-leg motors
   `[abad, hip, knee, foot(wheel)]` (flattened: wheels at indices 3/7/11/15). Verified **16/16 on the
   physical robot** by the passive hand-swing probe (`--probe`, 2026-10-15); signs/zeros additionally
   validated against factory lie/stand pose snapshots (all 12 leg joints agree within 0.08 rad).
   See `runtime/joint_map_v2.py` (self-testing, torch-free) and the raw<->urdf sign/offset table
   `S16/C16` in `run_loco_v2.py`.
2. **IMU quaternion is `(x, y, z, w)`** — the `lowlevel.h` comment says wxyz and is **wrong** (measured
   live: as xyzw the quaternion gravity matches the accelerometer to 1.3°, as wxyz it is off by 104°).
   The runtime refuses to run if the first-step gravity-vs-accelerometer cross-check fails (gate #12).

The stand-up is a **replay of the factory stand-up trajectory** (recorded with `--record` while the
factory controller drives): tuck the feet under the body, settle onto them, then rise level. A naive
straight-line ramp to the standing pose collapses the rear of this rear-heavy machine (poor rear-leg
leverage from the sprawled pose) — that failure is why the replay exists.

## Repository layout

```
README.md                      this page
runtime/
  run_loco_v2.py               main control loop (r19): state machine, 500 Hz sender, e-stops,
                               stand-up replay, on-robot gates (--check), mapping helpers
                               (--probe/--inspect/--jog/--wheel_test/--snapshot/--record)
  loco_runtime_v2.py           observation/action math (single source of truth): history buffer,
                               gravity from quat, profiles, ONNX/TorchScript backends; self-testing
  joint_map_v2.py              SDK <-> policy order mapping; self-testing
policy/
  model_4999.pt                v2.0-stage1 — the checkpoint deployed on hardware (stage1 profile)
  model_9997.pt                v2.2 stage1c standing repair — current best, also a stage1-profile policy
```

`policy/` tracks the raw rsl_rl checkpoints (load them via the torch backend, or use the ONNX exports
from the deploy package built by `scripts/package_deploy_v2.py` — gates G1-G5 incl. ONNX==actor parity —
which also ships `OBS_CONTRACT_V2.md`). Friction randomization is identical across all v2 checkpoints
(static 0.4-1.0 / dynamic 0.3-0.9, never changed); no friction-retuned variant exists — the on-record mu
sweep was evaluation-only. The stand-up recording `logs/factory_traj_standup.npz` is produced per robot
with `--record` and is expected next to the runtime at run time.

## Where it runs

| | |
|---|---|
| Board | the robot's motion-control board ("dog"): firefly RK3588, aarch64, Ubuntu 22.04, PREEMPT_RT kernel |
| Inference | **onnxruntime** (CPU EP; no torch on the board) — use `policy/*.onnx` |
| Rates | 50 Hz policy / **500 Hz** command refresh (the 1 ms spline daemon drops commands refreshed slower than ~10 of its cycles, so a 50 Hz direct write does NOT work) |
| Scheduling | SCHED_FIFO 40 pinned to cores 5,6 — never core 7 (vendor daemons live there) |

## Checkpoint profiles (contract §4.1)

| | legs (12) | wheels (4) | a_prev fed back |
|---|---|---|---|
| **stage1** (first on robot) | `p_des = q_default + scale*a`, kp 20 / kd 0.7 | `v_des=0, kp=0, kd=5.0` — **deploy-side brake, policy wheel outputs discarded** | raw 16-dim |
| stage2 | same | `v_des = 5*a[12+k], kp=0, kd=0.7` | raw 16-dim |

**The policy does not lock the wheels itself.** The stage1 brake must be dispatched every tick by this
loop; before handover a hand-pushed wheel must feel viscously damped — if it spins freely the brake is
not being written and handover is forbidden.

⚠ `WHEEL_SIGN` in `run_loco_v2.py` is a `+1` placeholder until gate #10 (`--wheel_test` + simulation
cross-check) is completed. stage1 is sign-independent (v=0); **stage2 is forbidden until the measured
signs are filled in.** The hand-spin probe suggests `(-1, +1, -1, +1)` in SDK order FR, FL, BR, BL —
treat as a hypothesis, not a fact.

## Operating procedure

```bash
# 0) software gates, both profiles, all must PASS (no shared memory touched):
python3 runtime/run_loco_v2.py --check --policy policy/policy_stage1_model_4999.onnx --profile stage1
# 1) module self-tests:
python3 runtime/joint_map_v2.py
python3 runtime/loco_runtime_v2.py --policy policy/policy_stage1_model_4999.onnx --profile stage1
# 2) dry-run (reads real sensors, writes nothing; also measures the sender rate):
python3 runtime/run_loco_v2.py --policy policy/policy_stage1_model_4999.onnx --profile stage1
# 3) live (hoisted first; e-stop within reach; factory controller auto-stopped/restored):
sudo python3 runtime/run_loco_v2.py --live --policy policy/policy_stage1_model_4999.onnx --profile stage1
```

Keys: `1` stand up (factory replay), `2` enter policy (pose-gated), `SPACE` e-stop to damping,
`c` zero the command, `r` save a dump, `x` exit and restore the factory stack; hold `w/s` fwd/back,
`a/d` strafe, `q/e` turn (release <=0.6 s auto-zeros; `--step_keys` for stepped increments).

Safety chain (all automatic): tip-over (projected gravity z > -0.5, ≈60°); sender watchdog (targets
>100 ms stale -> damping, refresh continues); IMU stream-stall detection (500 ms); torque e-stop with
**motor-protection semantics** (instant stop above `--tau_impact` 40 N*m; one joint above
`--tau_overload` 32 N*m for 600 ms continuous — normal walking/standing peaks of 28~30 pass); command
slew `--cmd_slew` 2 m/s² (a 1 m/s brake takes 0.5 s — step commands at speed pitch the nose down).

Standing behavior note: leg actions |a| ≈ 1.2-1.5 while standing are **normal** (gravity compensation
under kp=20, quantitatively matched to the measured knee loads). Judge the stand by joint **positions**
staying near `q_default`, never by |a|.

## When something looks wrong

Record 600 frames (`dump.npy`: 570-dim network input + 16-dim raw action, written automatically on the
first policy entry, or `r` on demand), copy it to the training machine and run
`scripts/verify_dump_v2.py` — one run separates "wiring wrong" from "network wrong" (obs-assembly
asserts + bit-level replay parity, expected ~1e-6). Record a dump on the FIRST live session, not after
something has already fallen over. Every automatic e-stop also writes `estop_*.json` with the trigger
frame (why/state/tau16/qd16/gravity/cmd).

Calibration tolerances (beyond these is outside the training distribution): gyro residual bias
< 0.03 rad/s; IMU attitude error < ~1.1°; joint zero offset < 0.29°; joint velocity scale error < 5%.

## Policy lineage (training side)

| version | checkpoint | status |
|---|---|---|
| v2.0-stage1 (StepFirst, wheels braked) | `model_4999.pt` | **deployed on hardware** (stand-up, 1 m/s walking) |
| v2.0-stage2 (Rolling, wheels by policy) | `model_9998.pt` | simulation-recommended; blocked on WHEEL_SIGN (gate #10) |
| v2.1 (attitude + livelier gait finetune) | `model_7498.pt` | motion-domain best; standing pose regressed |
| v2.2 (standing repair on v2.1) | `model_9997.pt` | standing best-ever (settled 0.282); next on-robot candidate |

Training configs, evaluations and every on-robot verdict are recorded in the project training ledger
(`钢镚L1W-v2训练台账`) and `docs/TRAINING_REDESIGN_V2.md` on the training machine.

Known limits (by design, documented for downstream users): pure-rolling top speed ≈1.97 m/s (wheel
actuator saturates at 28 rad/s); a hard brake dips the nose — that is trained behavior (the attitude
cost was deliberately off in v2.0), mitigated on hardware by the command slew; sim does not reproduce
the real near-flip (payload/CoM/friction all excluded experimentally — the residual gap is actuator
saturation transients, wheel-contact modeling and ZOH latency).
