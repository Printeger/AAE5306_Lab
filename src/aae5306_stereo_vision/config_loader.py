#!/usr/bin/env python3
"""Utility helpers for loading the stereo vision pipeline configuration."""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np
import rospy

DEFAULT_NAMESPACE = '/aae5306_stereo_vision'
_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class ConfigError(RuntimeError):
    """Raised when the configuration is missing required content."""


def _reshape_matrix(data: Iterable[float]) -> np.ndarray:
    values = list(float(x) for x in data)
    if len(values) != 16:
        raise ConfigError(
            f'Expected 16 values for a 4x4 transform, received {len(values)}'
        )
    return np.array(values, dtype=float).reshape((4, 4))


def _parse_calibration(calibration: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cameras_map: Dict[str, Dict[str, Any]] = {}
    for entry in calibration.get('cameras', []):
        camera_info = entry.get('camera', {})
        label = camera_info.get('label')
        if not label:
            continue

        intrinsics_data = camera_info.get('intrinsics', {}).get('data', [])
        intrinsics = {}
        if len(intrinsics_data) >= 4:
            intrinsics = {
                'fx': float(intrinsics_data[0]),
                'fy': float(intrinsics_data[1]),
                'cx': float(intrinsics_data[2]),
                'cy': float(intrinsics_data[3]),
            }

        distortion_params = (
            camera_info.get('distortion', {})
            .get('parameters', {})
            .get('data', [])
        )

        transform_data = entry.get('T_B_C', {}).get('data', [])
        transform = None
        translation = None
        if len(transform_data) == 16:
            transform = _reshape_matrix(transform_data)
            translation = transform[:3, 3].copy()

        cameras_map[label] = {
            'id': camera_info.get('id'),
            'label': label,
            'intrinsics': intrinsics,
            'intrinsics_raw': camera_info.get('intrinsics'),
            'distortion': camera_info.get('distortion'),
            'distortion_parameters': [float(x) for x in distortion_params],
            'image_height': camera_info.get('image_height'),
            'image_width': camera_info.get('image_width'),
            'transform': transform,
            'translation': translation,
        }

    return cameras_map


def _compute_baseline(
    cameras_map: Dict[str, Dict[str, Any]],
    explicit_baseline: Optional[float],
) -> Optional[float]:
    if explicit_baseline is not None:
        return float(explicit_baseline)

    if len(cameras_map) < 2:
        return None

    labels = list(cameras_map.keys())
    origin_label = labels[0]
    origin_translation = cameras_map[origin_label].get('translation')
    if origin_translation is None:
        return None

    for label in labels[1:]:
        translation = cameras_map[label].get('translation')
        if translation is None:
            continue
        delta = origin_translation - translation
        return float(np.linalg.norm(delta))

    return None


def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    calibration = raw.get('calibration', {})
    cameras_map = _parse_calibration(calibration)

    stereo_cfg = dict(raw.get('stereo', {}) or {})
    baseline = _compute_baseline(cameras_map, stereo_cfg.get('baseline'))
    if baseline is not None:
        stereo_cfg['baseline'] = baseline

    normalized = {
        'config_version': raw.get('config_version'),
        'description': raw.get('description'),
        'calibration': calibration,
        'cameras': cameras_map,
        'processing': raw.get('processing', {}),
        'topics': raw.get('topics', {}),
        'nodes': raw.get('nodes', {}),
        'frames': raw.get('frames', {}),
        'stereo': stereo_cfg,
    }
    return normalized


def _load_from_param_server(namespace: str) -> Dict[str, Any]:
    if not rospy.has_param(namespace):
        raise ConfigError(
            f"Configuration namespace '{namespace}' not found on the parameter server"
        )
    data = rospy.get_param(namespace)
    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected configuration under '{namespace}' to be a dictionary"
        )
    return data


def get_pipeline_config(namespace: str = DEFAULT_NAMESPACE) -> Dict[str, Any]:
    """Return the parsed pipeline configuration for the given namespace."""
    with _CACHE_LOCK:
        if namespace not in _CONFIG_CACHE:
            raw = _load_from_param_server(namespace)
            _CONFIG_CACHE[namespace] = _normalize(raw)
        config = _CONFIG_CACHE[namespace]
    return copy.deepcopy(config)


def get_node_block(
    config: Dict[str, Any],
    node_name: str,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    nodes = config.get('nodes', {})
    if node_name not in nodes:
        raise ConfigError(f"No configuration found for node '{node_name}'")

    node_block = nodes[node_name]
    if expected_type:
        node_type = node_block.get('type', expected_type)
        if node_type != expected_type:
            raise ConfigError(
                f"Node '{node_name}' expects type '{expected_type}', found '{node_type}'"
            )
    return node_block


def get_camera(config: Dict[str, Any], label: str) -> Dict[str, Any]:
    cameras = config.get('cameras', {})
    if label not in cameras:
        raise ConfigError(f"Camera '{label}' not defined in configuration")
    return cameras[label]


def require_path(config: Dict[str, Any], path: Iterable[str], what: str) -> Any:
    cursor: Any = config
    for key in path:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            pretty = '/'.join(path)
            raise ConfigError(f"Missing {what} at '{pretty}' in configuration")
    return cursor


def reset_cache() -> None:
    """Clear cached configuration so it can be reloaded."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()