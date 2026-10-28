# v2 joint mapping: vendor SDK order <-> policy order. Torch-free, copies to the robot as-is.
#
# FACT SOURCE (read, not assumed): /mnt/ssd2/genisom_l1/sdk/genisom_L1_sdk/include/zsibot_sdk/lowlevel/lowlevel.h
#   enum { LEG_FRONT_RIGHT = 0, LEG_FRONT_LEFT = 1, LEG_BACK_RIGHT = 2, LEG_BACK_LEFT = 3 };
#   leg_control_t / leg_state_t = { abad, hip, knee, foot }   // foot = wheel, 4th joint of every leg
# So the flattened 16-dim SDK vector is per-leg interleaved:
#   sdk[leg*4 + {0:abad, 1:hip, 2:knee, 3:foot}],  legs FR, FL, BR, BL  ->  wheels at [3, 7, 11, 15]
#
# Policy order (locked at training time with preserve_order=True, config genisom_l1w_v2):
#   [0:12]  legs, leg-grouped:  FAR(FR): abad,hip,knee | FBL(FL) | RAR(BR) | RBL(BL)
#   [12:16] wheels:             FAR, FBL, RAR, RBL
# Leg sequence and within-leg order MATCH the SDK, so the mapping is ONE structural reshape:
#   policy_legs = sdk.reshape(4,4)[:, :3].flatten()     |     policy_wheels = sdk.reshape(4,4)[:, 3]
#
# A = Right, B = Left (confirmed three ways: official URDF kinematic chain, asset-config naming note,
# and the wheel-spin experiment: FR wheel turned by hand accumulated 9.54 rad vs 1.65-1.78 on the others).
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

N_SDK = 16
WHEEL_SDK_INDICES = (3, 7, 11, 15)
LEG_SDK_INDICES = tuple(i for i in range(16) if i not in WHEEL_SDK_INDICES)

SDK_LEG_NAMES = ("FR", "FL", "BR", "BL")              # lowlevel.h enum order
SDK_JOINT_NAMES = tuple(
    f"{leg}_{j}" for leg in SDK_LEG_NAMES for j in ("abad", "hip", "knee", "foot")
)
POLICY_LEG_JOINT_NAMES = tuple(
    f"{leg}_{j}" for leg in ("FAR", "FBL", "RAR", "RBL") for j in ("ABAD", "HIP", "KNEE")
)
POLICY_WHEEL_NAMES = ("FAR_FOOT", "FBL_FOOT", "RAR_FOOT", "RBL_FOOT")

# asset name <-> SDK name (A=Right, B=Left)
_ASSET2SDK_LEG = {"FAR": "FR", "FBL": "FL", "RAR": "BR", "RBL": "BL"}


def sdk_to_policy(v16):
    """SDK-order 16 values -> (legs12, wheels4) in policy order. Works on any indexable sequence."""
    legs = [v16[leg * 4 + j] for leg in range(4) for j in range(3)]
    wheels = [v16[leg * 4 + 3] for leg in range(4)]
    return legs, wheels


def policy_to_sdk(legs12, wheels4):
    """(legs12, wheels4) in policy order -> SDK-order 16 values."""
    out = [0.0] * 16
    for leg in range(4):
        for j in range(3):
            out[leg * 4 + j] = legs12[leg * 3 + j]
        out[leg * 4 + 3] = wheels4[leg]
    return out


def _selftest() -> int:
    bad = 0
    print("SDK order     :", "  ".join(f"{i}:{n}" for i, n in enumerate(SDK_JOINT_NAMES)))
    print("policy legs   :", "  ".join(f"{i}:{n}" for i, n in enumerate(POLICY_LEG_JOINT_NAMES)))
    print("policy wheels :", "  ".join(f"{i+12}:{n}" for i, n in enumerate(POLICY_WHEEL_NAMES)))
    # 1. roundtrip bijection
    probe = list(range(100, 116))
    legs, wheels = sdk_to_policy(probe)
    if policy_to_sdk(legs, wheels) != probe:
        print("FAIL roundtrip mismatch")
        bad += 1
    # 2. wheels land where lowlevel.h says
    if [probe[i] for i in WHEEL_SDK_INDICES] != wheels:
        print("FAIL wheel indices are not [3,7,11,15]")
        bad += 1
    # 3. NAME-level agreement, not just indices
    for leg in range(4):
        for j, jname in enumerate(("ABAD", "HIP", "KNEE")):
            pol = POLICY_LEG_JOINT_NAMES[leg * 3 + j]                      # e.g. FBL_HIP
            sdk = SDK_JOINT_NAMES[leg * 4 + j]                             # e.g. FL_hip
            if _ASSET2SDK_LEG[pol.split("_")[0]] != sdk.split("_")[0] or jname.lower() != sdk.split("_")[1]:
                print(f"FAIL name mismatch: policy[{leg*3+j}]={pol} vs sdk[{leg*4+j}]={sdk}")
                bad += 1
        pol_w = POLICY_WHEEL_NAMES[leg]
        sdk_w = SDK_JOINT_NAMES[leg * 4 + 3]
        if _ASSET2SDK_LEG[pol_w.split("_")[0]] != sdk_w.split("_")[0]:
            print(f"FAIL wheel-name mismatch: {pol_w} vs {sdk_w}")
            bad += 1
    print("mapping self-test:", "OK" if bad == 0 else "FAILED")
    print("NOTE: this self-test cannot verify each wheel's **spin-direction sign** — on-robot gate #10: per wheel v_des=+1 rad/s,")
    print("the read-back qdot must be positive and the drive direction must match simulation (left/right URDF axes may be mirrored).")
    return bad


if __name__ == "__main__":
    import sys

    sys.exit(1 if _selftest() else 0)
