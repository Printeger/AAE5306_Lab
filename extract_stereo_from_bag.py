#!/usr/bin/env python3
"""
Extract synchronized stereo image pairs from a ROS bag.

Features:
- Reads left/right image topics from a rosbag.
- Maintains timestamp synchronization within a tolerance window.
- Saves every Nth synchronized pair (default: 10) as PNG.
- Outputs to /home/hu/aae5306_lab9_ws/data/image/left and /home/hu/aae5306_ws/data/image/right by default.

Example:
  python3 extract_stereo_from_bag.py \
      --bag /path/to/data.bag \
      --left-topic /cam0/image_raw --right-topic /cam1/image_raw \
      --output-root /home/hu/aae5306_ws/data/image \
      --stride 10 --tolerance 0.01 --max-pairs 280
"""

import os
import argparse
from collections import deque
from typing import Deque, Optional, Tuple

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def stamp_to_ns(img_msg: Image) -> int:
    stamp = img_msg.header.stamp
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


def stamp_to_sec(img_msg: Image) -> float:
    stamp = img_msg.header.stamp
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def try_match_pair(
    anchor_msg: Image,
    candidate_buffer: Deque[Image],
    tolerance_sec: float,
) -> Optional[Tuple[Image, Image]]:
    if not candidate_buffer:
        return None

    anchor_t = stamp_to_sec(anchor_msg)

    best_index = -1
    best_dt = 1e9
    for idx, cand in enumerate(candidate_buffer):
        dt = abs(stamp_to_sec(cand) - anchor_t)
        if dt < best_dt:
            best_dt = dt
            best_index = idx

    if best_dt <= tolerance_sec and best_index >= 0:
        cand = candidate_buffer[best_index]
        del candidate_buffer[best_index]
        return (anchor_msg, cand)

    # Drop stale candidates that are too old
    while candidate_buffer and (anchor_t - stamp_to_sec(candidate_buffer[0]) > tolerance_sec):
        candidate_buffer.popleft()

    return None


def save_png(image_msg: Image, out_path: str, bridge: CvBridge) -> None:
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    # Convert mono/grayscale to 8-bit if necessary for consistent PNG
    if cv_img.dtype != 'uint8' and cv_img.dtype != 'uint16':
        cv_img = cv2.convertScaleAbs(cv_img)
    cv2.imwrite(out_path, cv_img)


def extract(
    bag_path: str,
    left_topic: str,
    right_topic: str,
    output_root: str,
    stride: int,
    tolerance_sec: float,
    max_pairs: Optional[int] = None,
) -> None:
    left_dir = os.path.join(output_root, 'left')
    right_dir = os.path.join(output_root, 'right')
    ensure_dir(left_dir)
    ensure_dir(right_dir)

    bridge = CvBridge()

    left_buffer: Deque[Image] = deque()
    right_buffer: Deque[Image] = deque()

    paired_counter = 0
    saved_counter = 0

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=[left_topic, right_topic]):

            if topic == left_topic:
                # Try to match with any buffered right
                pair = try_match_pair(msg, right_buffer, tolerance_sec)
                if pair is None:
                    left_buffer.append(msg)
                else:
                    left_msg, right_msg = pair
                    paired_counter += 1
                    if paired_counter % stride == 0:
                        stamp_ns = stamp_to_ns(left_msg)
                        left_path = os.path.join(left_dir, f"{stamp_ns:019d}.png")
                        right_path = os.path.join(right_dir, f"{stamp_ns:019d}.png")
                        save_png(left_msg, left_path, bridge)
                        save_png(right_msg, right_path, bridge)
                        saved_counter += 1
                        if max_pairs is not None and saved_counter >= max_pairs:
                            break

            elif topic == right_topic:
                # Try to match with any buffered left
                pair = try_match_pair(msg, left_buffer, tolerance_sec)
                if pair is None:
                    right_buffer.append(msg)
                else:
                    right_msg, left_msg = pair
                    paired_counter += 1
                    if paired_counter % stride == 0:
                        stamp_ns = stamp_to_ns(left_msg)
                        left_path = os.path.join(left_dir, f"{stamp_ns:019d}.png")
                        right_path = os.path.join(right_dir, f"{stamp_ns:019d}.png")
                        save_png(left_msg, left_path, bridge)
                        save_png(right_msg, right_path, bridge)
                        saved_counter += 1
                        if max_pairs is not None and saved_counter >= max_pairs:
                            break

    print(f"Total synchronized pairs found: {paired_counter}")
    print(f"Pairs saved (every {stride}): {saved_counter}")
    print(f"Left images: {left_dir}")
    print(f"Right images: {right_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract synchronized stereo images from a rosbag")
    parser.add_argument('--bag', required=True, help='Path to rosbag file')
    parser.add_argument('--left-topic', default='/cam0/image_raw', help='Left camera topic')
    parser.add_argument('--right-topic', default='/cam1/image_raw', help='Right camera topic')
    parser.add_argument('--output-root', default='data/image', help='Output root directory containing left/right')
    parser.add_argument('--stride', type=int, default=10, help='Save every Nth synchronized pair')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Timestamp tolerance (seconds) for pairing')
    parser.add_argument('--max-pairs', type=int, default=280, help='Optional limit on number of saved pairs (<= 0 to disable)')

    args = parser.parse_args()
    max_pairs = args.max_pairs if args.max_pairs and args.max_pairs > 0 else None

    extract(
        bag_path=args.bag,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        output_root=args.output_root,
        stride=max(1, args.stride),
        tolerance_sec=max(0.0, args.tolerance),
        max_pairs=max_pairs,
    )


if __name__ == '__main__':
    main()


