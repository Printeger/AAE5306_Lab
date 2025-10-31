#!/usr/bin/env python3
"""
Extract stereo parameters from RealSense CameraInfo topics and write a
stereo_params YAML compatible with AAE5306 Stereo Vision config.

Defaults assume topics:
  - Left  CameraInfo:  /camera/infra1/camera_info
  - Right CameraInfo:  /camera/infra2/camera_info
  - Left  image:       /camera/infra1/image_rect_raw
  - Right image:       /camera/infra2/image_rect_raw

Usage example:
  rosrun AAE5306_Lab extract_realsense_stereo_params.py \
    --left-info /camera/infra1/camera_info \
    --right-info /camera/infra2/camera_info \
    --left-image /camera/infra1/image_rect_raw \
    --right-image /camera/infra2/image_rect_raw \
    --output stereo_params.yaml
"""

import argparse
import sys
from typing import Dict, Any, List

import rospy
from sensor_msgs.msg import CameraInfo

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _fmt_vec(data: List[float]) -> Dict[str, Any]:
    return {
        'cols': 1,
        'rows': len(data),
        'data': [float(x) for x in data],
    }


def _fmt_mat_4x4(data_4x4: List[List[float]]) -> Dict[str, Any]:
    flat: List[float] = []
    for r in data_4x4:
        flat.extend([float(x) for x in r])
    return {
        'cols': 4,
        'rows': 4,
        'data': flat,
    }


def _intrinsics_from_K(K: List[float]):
    # K is row-major 3x3 (length 9)
    fx = float(K[0])
    fy = float(K[4])
    cx = float(K[2])
    cy = float(K[5])
    return fx, fy, cx, cy


def _baseline_from_P(P_right: List[float]) -> float:
    # P is row-major 3x4 (length 12)
    fx = float(P_right[0])
    Tx_pix = float(P_right[3])
    # Standard rectified stereo: P_right[0,3] = -fx * baseline
    if fx == 0.0:
        return 0.0
    baseline = -Tx_pix / fx
    return float(baseline)


def _build_camera_block(label: str, info: CameraInfo, distortion_len: int = 4) -> Dict[str, Any]:
    fx, fy, cx, cy = _intrinsics_from_K(info.K)
    D = list(info.D) if info.D is not None else []
    if len(D) < distortion_len:
        D = D + [0.0] * (distortion_len - len(D))
    else:
        D = D[:distortion_len]

    return {
        'camera': {
            'label': label,
            'id': label,
            'line-delay-nanoseconds': 0,
            'image_height': int(info.height),
            'image_width': int(info.width),
            'type': 'pinhole',
            'intrinsics': _fmt_vec([fx, fy, cx, cy]),
            'distortion': {
                'type': 'radial-tangential',
                'parameters': _fmt_vec(D),
            },
        },
        # Filled later by caller if needed
        'T_B_C': _fmt_mat_4x4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    }


def _build_processing_defaults() -> Dict[str, Any]:
    return {
        'detector': {
            'type': 'orb',
            'publish_rate': 10.0,
            'visualize': True,
        },
        'matching': {
            'ratio_threshold': 0.75,
            'epipolar_threshold': 2.0,
            'max_delay': 0.1,
            'visualize': True,
        },
        'depth': {
            'min_depth': 0.5,
            'max_depth': 15.0,
            'min_ray_depth': 3.0,
            'max_ray_depth': 12.0,
            'max_delay': 0.1,
            'visualize': True,
        },
    }


def _build_topics(left_label: str, right_label: str,
                  left_image_topic: str, right_image_topic: str) -> Dict[str, Any]:
    return {
        'inputs': {
            left_label: left_image_topic,
            right_label: right_image_topic,
        },
        'outputs': {
            'feature_detection': {
                left_label: {
                    'features_image': '/stereo_vision/cam0/features_image' if left_label == 'cam0' else f'/stereo_vision/{left_label}/features_image',
                    'feature_stats': '/stereo_vision/cam0/feature_stats' if left_label == 'cam0' else f'/stereo_vision/{left_label}/feature_stats',
                },
                right_label: {
                    'features_image': '/stereo_vision/cam1/features_image' if right_label == 'cam1' else f'/stereo_vision/{right_label}/features_image',
                    'feature_stats': '/stereo_vision/cam1/feature_stats' if right_label == 'cam1' else f'/stereo_vision/{right_label}/feature_stats',
                },
            },
            'feature_matching': {
                'matches_image': '/stereo_vision/matches_image',
                'match_stats': '/stereo_vision/match_stats',
            },
            'depth_recovery': {
                'pointcloud': '/stereo_vision/pointcloud',
                'depth_image': '/stereo_vision/depth_image',
                'depth_stats': '/stereo_vision/depth_stats',
            },
        },
    }


def _build_frames(left_label: str) -> Dict[str, Any]:
    return {
        left_label: left_label,
        'cam1': 'cam1' if left_label == 'cam0' else 'cam1',
        'pointcloud': left_label,
    }


def _build_nodes(left_label: str, right_label: str) -> Dict[str, Any]:
    return {
        'feature_detection_cam0': {
            'type': 'feature_detection',
            'camera': left_label,
        },
        'feature_detection_cam1': {
            'type': 'feature_detection',
            'camera': right_label,
        },
        'feature_matching': {
            'type': 'feature_matching',
            'left_camera': left_label,
            'right_camera': right_label,
        },
        'depth_recovery': {
            'type': 'depth_recovery',
            'left_camera': left_label,
            'right_camera': right_label,
        },
    }


def extract(left_info_topic: str, right_info_topic: str,
            left_label: str = 'cam0', right_label: str = 'cam1',
            left_image_topic: str = '/camera/infra1/image_rect_raw',
            right_image_topic: str = '/camera/infra2/image_rect_raw') -> Dict[str, Any]:
    rospy.loginfo(f"Waiting for CameraInfo: left={left_info_topic}, right={right_info_topic}")
    left_info: CameraInfo = rospy.wait_for_message(left_info_topic, CameraInfo, timeout=10.0)
    right_info: CameraInfo = rospy.wait_for_message(right_info_topic, CameraInfo, timeout=10.0)

    # Compute stereo baseline from right projection matrix
    baseline = _baseline_from_P(right_info.P)
    rospy.loginfo(f"Estimated baseline: {baseline:.6f} m")

    left_block = _build_camera_block(left_label, left_info)
    right_block = _build_camera_block(right_label, right_info)

    # Assume body frame equals left camera frame: T_B_C(left)=I, T_B_C(right)=translation along x
    right_block['T_B_C'] = _fmt_mat_4x4([
        [1.0, 0.0, 0.0, baseline],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    processing = _build_processing_defaults()
    topics = _build_topics(left_label, right_label, left_image_topic, right_image_topic)
    frames = _build_frames(left_label)
    nodes = _build_nodes(left_label, right_label)

    cfg: Dict[str, Any] = {
        'aae5306_stereo_vision': {
            'config_version': 1,
            'description': 'Extracted from RealSense CameraInfo',
            'calibration': {
                'label': 'Intel RealSense (from CameraInfo)',
                'id': 'realsense',
                'cameras': [left_block, right_block],
            },
            'processing': processing,
            'topics': topics,
            'frames': frames,
            'nodes': nodes,
            'stereo': {
                'baseline': float(baseline),
            },
        }
    }
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Extract stereo params from RealSense CameraInfo topics')
    parser.add_argument('--left-info', default='/camera/infra1/camera_info', help='Left CameraInfo topic')
    parser.add_argument('--right-info', default='/camera/infra2/camera_info', help='Right CameraInfo topic')
    parser.add_argument('--left-image', default='/camera/infra1/image_rect_raw', help='Left image topic for inputs mapping')
    parser.add_argument('--right-image', default='/camera/infra2/image_rect_raw', help='Right image topic for inputs mapping')
    parser.add_argument('--left-label', default='cam0', help='Left camera label to use in YAML')
    parser.add_argument('--right-label', default='cam1', help='Right camera label to use in YAML')
    parser.add_argument('-o', '--output', default='stereo_params.yaml', help='Output YAML file path')

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node('extract_realsense_stereo_params', anonymous=True, disable_signals=True)

    try:
        cfg = extract(
            args.left_info,
            args.right_info,
            args.left_label,
            args.right_label,
            args.left_image,
            args.right_image,
        )
    except rospy.ROSException as e:
        rospy.logerr(f"Failed to read CameraInfo topics: {e}")
        sys.exit(2)

    if yaml is None:
        rospy.logwarn("PyYAML not available; printing Python dict to stdout instead of YAML")
        print(cfg)
        return

    with open(args.output, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    rospy.loginfo(f"Wrote stereo parameters to: {args.output}")


if __name__ == '__main__':
    main()
