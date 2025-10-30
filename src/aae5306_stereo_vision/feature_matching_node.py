#!/usr/bin/env python3
"""
Feature Matching ROS Node

This node subscribes to synchronized stereo image pairs, performs feature matching,
and publishes match results with visualization.

Author: AAE5306 Teaching Team
Date: October 30, 2025
"""

import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber
from aae5306_stereo_vision.msg import MatchStats

# Minimal inline config loader to keep the project simple
import threading
from typing import Any, Dict, Optional

DEFAULT_NAMESPACE = '/aae5306_stereo_vision'
_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class ConfigError(RuntimeError):
    pass


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
            _CONFIG_CACHE[namespace] = _load_from_param_server(namespace)
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


def check_sift_available():
    """Check if SIFT is available in OpenCV."""
    return hasattr(cv2, 'SIFT_create')


class FeatureMatcher:
    """Match features between stereo image pairs."""
    
    def __init__(self, detector_type='sift', ratio_threshold=0.75, epipolar_threshold=2.0):
        """
        Initialize feature matcher.
        
        Args:
            detector_type (str): Type of detector ('sift' or 'orb')
            ratio_threshold (float): Lowe's ratio test threshold
            epipolar_threshold (float): Maximum y-coordinate difference (pixels)
        """
        self.detector_type = detector_type.lower()
        self.ratio_threshold = ratio_threshold
        self.epipolar_threshold = epipolar_threshold
        
        self.detector = None
        self.matcher = None
        self._initialize()
    
    def _initialize(self):
        """Initialize detector and matcher."""
        if self.detector_type == 'sift':
            if not check_sift_available():
                rospy.logwarn("SIFT not available in OpenCV. Falling back to ORB.")
                rospy.logwarn("To enable SIFT: pip3 install opencv-contrib-python")
                self.detector_type = 'orb'
                self.detector = cv2.ORB_create(nfeatures=2000)
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
                self.detector = cv2.SIFT_create()
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        elif self.detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
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
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def detect_and_match(self, img_left, img_right):
        """
        Detect features and match between stereo pair.
        
        Args:
            img_left (np.array): Left image (grayscale)
            img_right (np.array): Right image (grayscale)
            
        Returns:
            dict: Matching results with statistics and keypoints
        """
        # Detect features
        start_time = time.time()
        kp_left, desc_left = self.detector.detectAndCompute(img_left, None)
        kp_right, desc_right = self.detector.detectAndCompute(img_right, None)
        detection_time = (time.time() - start_time) * 1000
        
        if desc_left is None or desc_right is None or len(desc_left) < 2 or len(desc_right) < 2:
            return self._empty_result()
        
        # Match features
        start_time = time.time()
        matches = self.matcher.knnMatch(desc_left, desc_right, k=2)
        matching_time = (time.time() - start_time) * 1000
        
        # Apply ratio test
        start_time = time.time()
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # Apply epipolar constraint
        filtered_matches = []
        for match in good_matches:
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            
            if abs(pt_left[1] - pt_right[1]) < self.epipolar_threshold:
                filtered_matches.append(match)
        
        filtering_time = (time.time() - start_time) * 1000
        
        return {
            'keypoints_left': kp_left,
            'keypoints_right': kp_right,
            'matches': filtered_matches,
            'num_initial': len(matches),
            'num_ratio_filtered': len(good_matches),
            'num_final': len(filtered_matches),
            'detection_time': detection_time,
            'matching_time': matching_time,
            'filtering_time': filtering_time
        }
    
    def _empty_result(self):
        """Return empty result when matching fails."""
        return {
            'keypoints_left': [],
            'keypoints_right': [],
            'matches': [],
            'num_initial': 0,
            'num_ratio_filtered': 0,
            'num_final': 0,
            'detection_time': 0,
            'matching_time': 0,
            'filtering_time': 0
        }


class FeatureMatchingNode:
    """ROS node for real-time feature matching."""
    
    def __init__(self):
        """Initialize the feature matching node."""
        rospy.init_node('feature_matching_node', anonymous=False)

        try:
            self._initialize_from_config()
        except ConfigError as exc:
            rospy.logfatal(f"Configuration error: {exc}")
            raise

        # Initialize matcher
        self.matcher = FeatureMatcher(
            self.detector_type,
            self.ratio_threshold,
            self.epipolar_threshold
        )
        self.bridge = CvBridge()

        # Publishers
        self.match_viz_pub = rospy.Publisher(
            self.matches_image_topic, Image, queue_size=5
        )
        self.stats_pub = rospy.Publisher(
            self.match_stats_topic, MatchStats, queue_size=10
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

        rospy.loginfo("Feature Matching Node initialized")
        rospy.loginfo(f"  Detector type: {self.detector_type}")
        rospy.loginfo(f"  Ratio threshold: {self.ratio_threshold}")
        rospy.loginfo(f"  Epipolar threshold: {self.epipolar_threshold}")
        rospy.loginfo(f"  Left camera/topic: {self.left_camera_label} ({self.left_topic})")
        rospy.loginfo(f"  Right camera/topic: {self.right_camera_label} ({self.right_topic})")
        rospy.loginfo(f"  Visualization: {self.visualize}")
        if self.config_version is not None:
            rospy.loginfo(f"  Config version: {self.config_version}")
    
    def _initialize_from_config(self):
        """Load matching configuration from the shared YAML file."""
        self.node_name = rospy.get_name().split('/')[-1]
        self.config = get_pipeline_config()
        self.config_version = self.config.get('config_version')

        node_block = get_node_block(
            self.config, self.node_name, expected_type='feature_matching'
        )

        self.left_camera_label = node_block.get('left_camera')
        self.right_camera_label = node_block.get('right_camera')
        if not self.left_camera_label or not self.right_camera_label:
            raise ConfigError(
                f"Node '{self.node_name}' must define 'left_camera' and 'right_camera'"
            )

        detector_cfg = self.config.get('processing', {}).get('detector', {})
        detector_type = detector_cfg.get('type', 'orb')
        self.detector_type = str(detector_type)

        matching_cfg = self.config.get('processing', {}).get('matching', {})
        ratio = matching_cfg.get('ratio_threshold', 0.75)
        epipolar = matching_cfg.get('epipolar_threshold', 2.0)
        delay = matching_cfg.get('max_delay', 0.1)

        self.ratio_threshold = float(ratio)
        self.epipolar_threshold = float(epipolar)
        self.max_delay = float(delay)

        visualize_default = matching_cfg.get('visualize', True)
        self.visualize = bool(node_block.get('visualize', visualize_default))

        topics_cfg = self.config.get('topics', {})
        inputs_cfg = topics_cfg.get('inputs', {})
        for camera in (self.left_camera_label, self.right_camera_label):
            if camera not in inputs_cfg:
                raise ConfigError(
                    f"Input topic for camera '{camera}' not defined"
                )
        self.left_topic = inputs_cfg[self.left_camera_label]
        self.right_topic = inputs_cfg[self.right_camera_label]

        outputs_cfg = topics_cfg.get('outputs', {}).get('feature_matching', {})
        self.matches_image_topic = outputs_cfg.get('matches_image')
        self.match_stats_topic = outputs_cfg.get('match_stats')

        if not self.matches_image_topic or not self.match_stats_topic:
            raise ConfigError(
                "Feature matching output topics are not fully defined"
            )

    def stereo_callback(self, left_msg, right_msg):
        """Process synchronized stereo pair."""
        try:
            # Convert ROS Images to OpenCV
            img_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            img_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
            
            # Perform matching
            result = self.matcher.detect_and_match(img_left, img_right)
            
            # Publish statistics
            self.publish_stats(left_msg.header, result)
            
            # Publish visualization
            if self.visualize and self.match_viz_pub.get_num_connections() > 0:
                self.publish_visualization(
                    left_msg.header, img_left, img_right, result
                )
            
            rospy.loginfo_throttle(
                2.0,
                f"Matched {result['num_final']} features "
                f"({result['detection_time'] + result['matching_time']:.1f}ms)"
            )
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing stereo pair: {e}")
    
    def publish_stats(self, header, result):
        """Publish matching statistics."""
        stats = MatchStats()
        stats.header = header
        stats.detector_type = self.detector_type
        stats.num_initial_matches = result['num_initial']
        stats.num_ratio_filtered = result['num_ratio_filtered']
        stats.num_epipolar_filtered = result['num_final']
        stats.num_final_matches = result['num_final']
        stats.detection_time = result['detection_time']
        stats.matching_time = result['matching_time']
        stats.filtering_time = result['filtering_time']
        stats.total_time = (
            result['detection_time'] +
            result['matching_time'] +
            result['filtering_time']
        )
        stats.ratio_threshold = self.ratio_threshold
        stats.epipolar_threshold = self.epipolar_threshold
        
        self.stats_pub.publish(stats)
    
    def publish_visualization(self, header, img_left, img_right, result):
        """Publish visualization with matched features."""
        # Draw matches
        match_img = cv2.drawMatches(
            img_left, result['keypoints_left'],
            img_right, result['keypoints_right'],
            result['matches'], None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add detailed text overlays with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Statistics
        text1 = f"Detector: {self.detector_type.upper()}"
        text2 = f"Initial Matches: {result['num_initial']}"
        text3 = f"Ratio Test: {result['num_ratio_filtered']}"
        text4 = f"Epipolar Filter: {result['num_final']}"
        total_time = result['detection_time'] + result['matching_time'] + result['filtering_time']
        text5 = f"Time: {total_time:.1f}ms"
        
        # Draw text with black background for better visibility
        y_offset = 25
        for i, text in enumerate([text1, text2, text3, text4, text5]):
            y = y_offset + i * 30
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(match_img, (5, y - text_h - 5), (15 + text_w, y + 5), (0, 0, 0), -1)
            cv2.putText(match_img, text, (10, y), font, font_scale, (0, 255, 0), thickness)
        
        # Convert to ROS Image
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(match_img, encoding='bgr8')
            viz_msg.header = header
            self.match_viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish visualization: {e}")
    
    def run(self):
        """Keep the node running."""
        rospy.spin()


def main():
    """Main entry point."""
    try:
        node = FeatureMatchingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()
