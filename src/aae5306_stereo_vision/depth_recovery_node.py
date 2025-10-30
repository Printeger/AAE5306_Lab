#!/usr/bin/env python3
"""
Depth Recovery ROS Node

This node subscribes to synchronized stereo images, performs feature matching,
computes depth, and publishes 3D point clouds.

Author: AAE5306 Teaching Team
Date: October 30, 2025
"""

import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber
from aae5306_stereo_vision.msg import DepthStats

# Minimal inline config loader to keep the project simple
import threading
from typing import Any, Dict, Iterable, Optional

DEFAULT_NAMESPACE = '/aae5306_stereo_vision'
_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class ConfigError(RuntimeError):
    pass


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
        transform_data = entry.get('T_B_C', {}).get('data', [])
        transform = None
        translation = None
        if len(transform_data) == 16:
            transform = _reshape_matrix(transform_data)
            translation = transform[:3, 3].copy()
        cameras_map[label] = {
            'intrinsics': intrinsics,
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
    return {
        'config_version': raw.get('config_version'),
        'cameras': cameras_map,
        'processing': raw.get('processing', {}),
        'topics': raw.get('topics', {}),
        'nodes': raw.get('nodes', {}),
        'frames': raw.get('frames', {}),
        'stereo': stereo_cfg,
    }


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
    with _CACHE_LOCK:
        if namespace not in _CONFIG_CACHE:
            raw = _load_from_param_server(namespace)
            _CONFIG_CACHE[namespace] = _normalize(raw)
        config = _CONFIG_CACHE[namespace]
    return dict(config)


def get_node_block(
    config: Dict[str, Any], node_name: str, expected_type: Optional[str] = None
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
import sensor_msgs.point_cloud2 as pc2


def check_sift_available():
    """Check if SIFT is available in OpenCV."""
    return hasattr(cv2, 'SIFT_create')


class DepthRecovery:
    """Recover depth from stereo disparity."""
    
    def __init__(self, fx, fy, cx, cy, baseline):
        """
        Initialize depth recovery with camera calibration.
        
        Args:
            fx (float): Focal length in x (pixels)
            fy (float): Focal length in y (pixels)
            cx (float): Principal point x (pixels)
            cy (float): Principal point y (pixels)
            baseline (float): Stereo baseline (meters)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        
        rospy.loginfo(f"Depth Recovery Calibration:")
        rospy.loginfo(f"  fx={fx:.2f}, fy={fy:.2f}")
        rospy.loginfo(f"  cx={cx:.2f}, cy={cy:.2f}")
        rospy.loginfo(f"  baseline={baseline:.6f}m ({baseline*1000:.2f}mm)")
    
    def compute_depth(self, pts_left, pts_right, colors=None, min_depth=0.5, max_depth=15.0):
        """
        Compute depth from matched points.
        
        Args:
            pts_left (np.array): Matched points in left image (Nx2)
            pts_right (np.array): Matched points in right image (Nx2)
            colors (np.array): RGB colors for each point (Nx3), optional
            min_depth (float): Minimum valid depth (meters)
            max_depth (float): Maximum valid depth (meters)
            
        Returns:
            dict: Dictionary containing 3D points and statistics
        """
        # Compute disparity
        disparities = pts_left[:, 0] - pts_right[:, 0]
        
        # Filter invalid disparities
        valid_mask = disparities > 0
        
        pts_left_valid = pts_left[valid_mask]
        pts_right_valid = pts_right[valid_mask]
        disparities_valid = disparities[valid_mask]
        
        if colors is not None:
            colors_valid = colors[valid_mask]
        else:
            colors_valid = None
        
        # Compute depth: Z = (fx * baseline) / disparity
        depths = (self.fx * self.baseline) / disparities_valid
        
        # Filter by depth range
        depth_mask = (depths >= min_depth) & (depths <= max_depth)
        
        depths_filtered = depths[depth_mask]
        pts_left_filtered = pts_left_valid[depth_mask]
        disparities_filtered = disparities_valid[depth_mask]
        
        if colors_valid is not None:
            colors_filtered = colors_valid[depth_mask]
        else:
            colors_filtered = None
        
        # Compute 3D coordinates
        points_3d = self._compute_3d_points(pts_left_filtered, depths_filtered)
        
        # Compute statistics
        stats = {
            'num_points': len(points_3d),
            'num_valid_disparities': len(disparities_valid),
            'mean_depth': np.mean(depths_filtered) if len(depths_filtered) > 0 else 0,
            'min_depth': np.min(depths_filtered) if len(depths_filtered) > 0 else 0,
            'max_depth': np.max(depths_filtered) if len(depths_filtered) > 0 else 0,
            'std_depth': np.std(depths_filtered) if len(depths_filtered) > 0 else 0,
            'mean_disparity': np.mean(disparities_filtered) if len(disparities_filtered) > 0 else 0,
            'min_disparity': np.min(disparities_filtered) if len(disparities_filtered) > 0 else 0,
            'max_disparity': np.max(disparities_filtered) if len(disparities_filtered) > 0 else 0,
        }
        
        return {
            'points_3d': points_3d,
            'colors': colors_filtered,
            'depths': depths_filtered,
            'disparities': disparities_filtered,
            'stats': stats
        }
    
    def _compute_3d_points(self, pts_2d, depths):
        """
        Convert 2D image points to 3D camera coordinates.
        
        Args:
            pts_2d (np.array): 2D points (Nx2)
            depths (np.array): Depth values (N,)
            
        Returns:
            np.array: 3D points (Nx3)
        """
        points_3d = np.zeros((len(pts_2d), 3))
        
        # X = (u - cx) * Z / fx
        points_3d[:, 0] = (pts_2d[:, 0] - self.cx) * depths / self.fx
        
        # Y = (v - cy) * Z / fy
        points_3d[:, 1] = (pts_2d[:, 1] - self.cy) * depths / self.fy
        
        # Z = depth
        points_3d[:, 2] = depths
        
        return points_3d


class DepthRecoveryNode:
    """ROS node for real-time depth recovery."""
    
    def __init__(self):
        """Initialize the depth recovery node."""
        rospy.init_node('depth_recovery_node', anonymous=False)
        
        try:
            self._initialize_from_config()
        except ConfigError as exc:
            rospy.logfatal(f"Configuration error: {exc}")
            raise

        # Initialize components
        self.depth_recovery = DepthRecovery(self.fx, self.fy, self.cx, self.cy, self.baseline)
        self.bridge = CvBridge()
        
        # Initialize feature matcher
        if self.detector_type == 'sift':
            if not check_sift_available():
                rospy.logwarn("SIFT not available in OpenCV. Falling back to ORB.")
                rospy.logwarn("To enable SIFT: pip3 install opencv-contrib-python")
                self.detector_type = 'orb'
                self.detector = cv2.ORB_create(nfeatures=2000)
            else:
                self.detector = cv2.SIFT_create()
        elif self.detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
        
        # Initialize FLANN matcher
        if self.detector_type == 'sift':
            if check_sift_available():
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # ORB fallback
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Publishers
        self.pointcloud_pub = rospy.Publisher(
            self.pointcloud_topic, PointCloud2, queue_size=5
        )
        self.depth_viz_pub = rospy.Publisher(
            self.depth_image_topic, Image, queue_size=5
        )
        self.stats_pub = rospy.Publisher(
            self.depth_stats_topic, DepthStats, queue_size=10
        )
        
        # Synchronized subscribers
        self.left_sub = Subscriber(self.left_topic, Image)
        self.right_sub = Subscriber(self.right_topic, Image)
        
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=self.max_delay
        )
        self.sync.registerCallback(self.stereo_callback)
        
        rospy.loginfo("Depth Recovery Node initialized")
        rospy.loginfo(f"  Detector type: {self.detector_type}")
        rospy.loginfo(f"  Cameras: {self.left_camera_label} (left) | {self.right_camera_label} (right)")
        rospy.loginfo(f"  Topics: {self.left_topic} / {self.right_topic}")
        rospy.loginfo(f"  Matching thresholds: ratio={self.ratio_threshold}, epipolar={self.epipolar_threshold}")
        rospy.loginfo(f"  Depth range: {self.min_depth}m - {self.max_depth}m")
        rospy.loginfo(f"  Baseline: {self.baseline:.6f} m")
        rospy.loginfo(f"  Visualization: {self.visualize}")
        if self.config_version is not None:
            rospy.loginfo(f"  Config version: {self.config_version}")

    def _initialize_from_config(self):
        """Load depth recovery configuration from the shared YAML file."""
        self.node_name = rospy.get_name().split('/')[-1]
        self.config = config_loader.get_pipeline_config()
        self.config_version = self.config.get('config_version')

        node_block = get_node_block(
            self.config, self.node_name, expected_type='depth_recovery'
        )

        self.left_camera_label = node_block.get('left_camera')
        self.right_camera_label = node_block.get('right_camera')
        if not self.left_camera_label or not self.right_camera_label:
            raise config_loader.ConfigError(
                f"Node '{self.node_name}' must define 'left_camera' and 'right_camera'"
            )

        detector_cfg = self.config.get('processing', {}).get('detector', {})
        self.detector_type = str(detector_cfg.get('type', 'orb'))

        matching_cfg = self.config.get('processing', {}).get('matching', {})
        depth_cfg = self.config.get('processing', {}).get('depth', {})

        ratio = matching_cfg.get('ratio_threshold', 0.75)
        epipolar = matching_cfg.get('epipolar_threshold', 2.0)
        self.ratio_threshold = float(ratio)
        self.epipolar_threshold = float(epipolar)

        depth_min = depth_cfg.get('min_depth', 0.5)
        depth_max = depth_cfg.get('max_depth', 15.0)
        self.min_depth = float(depth_min)
        self.max_depth = float(depth_max)

        delay_source = depth_cfg.get('max_delay', matching_cfg.get('max_delay', 0.1))
        self.max_delay = float(delay_source)

        visualize_default = depth_cfg.get('visualize', True)
        self.visualize = bool(node_block.get('visualize', visualize_default))

        topics_cfg = self.config.get('topics', {})
        inputs_cfg = topics_cfg.get('inputs', {})
        for camera in (self.left_camera_label, self.right_camera_label):
            if camera not in inputs_cfg:
                raise config_loader.ConfigError(
                    f"Input topic for camera '{camera}' not defined"
                )
        self.left_topic = inputs_cfg[self.left_camera_label]
        self.right_topic = inputs_cfg[self.right_camera_label]

        outputs_cfg = topics_cfg.get('outputs', {}).get('depth_recovery', {})
        self.pointcloud_topic = outputs_cfg.get('pointcloud')
        self.depth_image_topic = outputs_cfg.get('depth_image')
        self.depth_stats_topic = outputs_cfg.get('depth_stats')
        if not all([self.pointcloud_topic, self.depth_image_topic, self.depth_stats_topic]):
            raise config_loader.ConfigError(
                "Depth recovery output topics are not fully defined"
            )

        camera = get_camera(self.config, self.left_camera_label)
        intrinsics = camera.get('intrinsics', {})
        try:
            self.fx = float(intrinsics['fx'])
            self.fy = float(intrinsics['fy'])
            self.cx = float(intrinsics['cx'])
            self.cy = float(intrinsics['cy'])
        except KeyError as exc:
            raise config_loader.ConfigError(
                f"Missing intrinsic '{exc.args[0]}' for camera '{self.left_camera_label}'"
            )

        baseline = self.config.get('stereo', {}).get('baseline')
        if baseline is None:
            raise config_loader.ConfigError('Stereo baseline is not defined')
        self.baseline = float(baseline)

        frames_cfg = self.config.get('frames', {})
        self.pointcloud_frame = str(frames_cfg.get('pointcloud', self.left_camera_label))

    
    def stereo_callback(self, left_msg, right_msg):
        """Process synchronized stereo pair."""
        try:
            start_time = time.time()
            
            # Convert ROS Images to OpenCV
            img_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            img_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
            
            # Detect and match features
            kp_left, desc_left = self.detector.detectAndCompute(img_left, None)
            kp_right, desc_right = self.detector.detectAndCompute(img_right, None)
            
            if desc_left is None or desc_right is None or len(desc_left) < 2 or len(desc_right) < 2:
                rospy.logwarn_throttle(5.0, "Insufficient features detected")
                return
            
            # Match features
            matches = self.matcher.knnMatch(desc_left, desc_right, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Apply epipolar constraint and extract points
            pts_left = []
            pts_right = []
            for match in good_matches:
                pt_left = kp_left[match.queryIdx].pt
                pt_right = kp_right[match.trainIdx].pt
                
                if abs(pt_left[1] - pt_right[1]) < self.epipolar_threshold:
                    pts_left.append(pt_left)
                    pts_right.append(pt_right)
            
            if len(pts_left) < 10:
                rospy.logwarn_throttle(5.0, f"Too few matches: {len(pts_left)}")
                return
            
            pts_left = np.array(pts_left, dtype=np.float32)
            pts_right = np.array(pts_right, dtype=np.float32)
            
            # Compute depth
            result = self.depth_recovery.compute_depth(
                pts_left, pts_right,
                min_depth=self.min_depth,
                max_depth=self.max_depth
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            # Publish point cloud
            if result['points_3d'].shape[0] > 0:
                self.publish_pointcloud(left_msg.header, result['points_3d'])
            
            # Publish depth visualization
            if self.visualize and self.depth_viz_pub.get_num_connections() > 0:
                self.publish_depth_visualization(
                    left_msg.header, img_left, pts_left, result['depths']
                )
            
            # Publish statistics
            self.publish_stats(left_msg.header, result['stats'], computation_time)
            
            rospy.loginfo_throttle(
                2.0,
                f"Generated {result['points_3d'].shape[0]} 3D points, "
                f"mean depth: {result['stats']['mean_depth']:.2f}m "
                f"({computation_time:.1f}ms)"
            )
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing stereo pair: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def publish_pointcloud(self, header, points_3d):
        """Publish 3D point cloud."""
        # Create PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        # Create point cloud message
        pc_msg = pc2.create_cloud(header, fields, points_3d)
        pc_msg.header.frame_id = self.pointcloud_frame

        self.pointcloud_pub.publish(pc_msg)
    
    def publish_depth_visualization(self, header, image, points, depths):
        """Publish depth visualization."""
        # Create colored depth image
        viz_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Normalize depths for visualization
        if len(depths) > 0:
            depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
            
            # Draw points with color based on depth
            for pt, depth_norm in zip(points, depths_norm):
                color = self._depth_to_color(depth_norm)
                cv2.circle(viz_image, (int(pt[0]), int(pt[1])), 3, color, -1)
        
        # Convert to ROS Image
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
            viz_msg.header = header
            self.depth_viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish depth visualization: {e}")
    
    def _depth_to_color(self, depth_norm):
        """Convert normalized depth to color (blue=close, red=far)."""
        # Use HSV color map
        hue = int((1.0 - depth_norm) * 120)  # Blue (120) to Red (0)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(x) for x in bgr[0, 0])
    
    def publish_stats(self, header, stats, computation_time):
        """Publish depth statistics."""
        msg = DepthStats()
        msg.header = header
        msg.num_points = stats['num_points']
        msg.num_valid_disparities = stats['num_valid_disparities']
        msg.mean_depth = stats['mean_depth']
        msg.min_depth = stats['min_depth']
        msg.max_depth = stats['max_depth']
        msg.std_depth = stats['std_depth']
        msg.mean_disparity = stats['mean_disparity']
        msg.min_disparity = stats['min_disparity']
        msg.max_disparity = stats['max_disparity']
        msg.focal_length = self.fx
        msg.baseline = self.baseline
        msg.computation_time = computation_time
        
        self.stats_pub.publish(msg)
    
    def run(self):
        """Keep the node running."""
        rospy.spin()


def main():
    """Main entry point."""
    try:
        node = DepthRecoveryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()
