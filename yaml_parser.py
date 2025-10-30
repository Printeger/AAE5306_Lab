#!/usr/bin/env python3
import sys
import yaml
import numpy as np

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def to_mat44(block):
    """Convert {'rows':4,'cols':4,'data':[...]} to 4x4 np.ndarray."""
    arr = np.array(block['data'], dtype=float).reshape(4, 4)
    return arr

def print_camera_info(cam_entry, name):
    cam = cam_entry["camera"]

    print(f"\n=== {name} ===")
    print(f"Image Size      : {cam['image_width']} x {cam['image_height']}")
    print(f"Camera Model    : {cam['type']}")

    fx, fy, cx, cy = cam["intrinsics"]["data"]
    print(f"Intrinsics      : fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")

    dist = cam["distortion"]
    print(f"Distortion      : {dist['type']}")
    print(f"Dist. Params    : {dist['parameters']['data']}")

    T = to_mat44(cam_entry["T_B_C"])
    print(f"T_B_C (4x4)     :\n{T}")

    return T

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 print_stereo_cams_full.py <stereo_params.yaml>")
        sys.exit(1)

    cfg = load_yaml(sys.argv[1])
    cams = cfg["aae5306_stereo_vision"]["calibration"]["cameras"]

    print("\n================= Stereo Camera Configuration =================")

    T0 = print_camera_info(cams[0], "cam0")
    T1 = print_camera_info(cams[1], "cam1")

    # Compute extrinsics cam0 → cam1
    T_C0_B = np.linalg.inv(T0)
    T_C0_C1 = T_C0_B @ T1
    tx, ty, tz = T_C0_C1[0, 3], T_C0_C1[1, 3], T_C0_C1[2, 3]
    baseline = abs(tx) * 1000  # mm

    print("\n=== cam0 → cam1 Relative Pose ===")
    print(f"T_C0_C1 (4x4)   :\n{T_C0_C1}")
    print(f"Translation     : tx={tx:.6f} m, ty={ty:.6f} m, tz={tz:.6f} m")
    print(f"Baseline        : {baseline:.2f} mm")

    print("\n===============================================================\n")


if __name__ == "__main__":
    main()
