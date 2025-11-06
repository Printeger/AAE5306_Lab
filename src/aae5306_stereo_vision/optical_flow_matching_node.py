#!/usr/bin/env python3
"""
Optical Flow Matching ROS Node

This node subscribes to synchronized stereo image pairs, performs FAST feature detection
and Lucas-Kanade optical flow for feature matching between stereo pairs and across frames,
similar to VINS-Mono approach.

Author: AAE5306 Teaching Team
Date: November 6, 2025
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
from typing import Any, Dict, Optional, Tuple, List

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


class OpticalFlowMatcher:
    """Match features using FAST detection and Lucas-Kanade optical flow."""
    
    def __init__(self, epipolar_threshold=2.0, lk_params=None, use_ransac=True, use_flow_back=True, max_features=100, min_features=80):
        """
        Initialize optical flow matcher.
        
        Args:
            epipolar_threshold (float): Maximum y-coordinate difference (pixels)
            lk_params (dict): Parameters for Lucas-Kanade optical flow
            use_ransac (bool): Whether to use RANSAC for geometric verification
            use_flow_back (bool): Whether to use forward-backward flow consistency check
            max_features (int): Maximum number of features to extract
            min_features (int): Minimum number of features to maintain for tracking
        """
        self.epipolar_threshold = epipolar_threshold
        self.use_ransac = use_ransac
        self.use_flow_back = use_flow_back
        self.max_features = max_features
        self.min_features = min_features
        
        # FAST detector
        self.fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
        
        # Lucas-Kanade parameters
        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # For temporal tracking (across frames)
        self.prev_left_img = None
        self.trackable_points = None  # Persistent points for temporal tracking
        self.temporal_tracks = []  # List of track histories [(x,y), (x,y), ...]
        
        # For stereo matching
        self.prev_right_img = None
    
    def detect_and_match(self, img_left, img_right):
        """
        Detect features with FAST and match using optical flow.
        
        Args:
            img_left (np.array): Left image (grayscale)
            img_right (np.array): Right image (grayscale)
            
        Returns:
            dict: Matching results with statistics and keypoints
        """
        start_time = time.time()
        
        # Detect features in left image
        kp_left = self.fast.detect(img_left, None)
        
        # Limit number of features if specified
        # if len(kp_left) > self.max_features:
        #     # Sort by response (strength) and take top features
        #     kp_left = sorted(kp_left, key=lambda kp: kp.response, reverse=True)[:self.max_features]
        
        detection_time = (time.time() - start_time) * 1000
        
        if not kp_left:
            return self._empty_result()
        
        # Convert keypoints to points for optical flow
        p0_left = np.array([kp.pt for kp in kp_left], dtype=np.float32).reshape(-1, 1, 2)
        
        # Stereo matching: Track from left to right using optical flow
        start_time = time.time()
        p1_right, st_right, err_right = cv2.calcOpticalFlowPyrLK(
            img_left, img_right, p0_left, None, **self.lk_params
        )
        stereo_matching_time = (time.time() - start_time) * 1000
        
        # Forward-backward consistency check if enabled
        if self.use_flow_back:
            start_time = time.time()
            p1_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
                img_right, img_left, p1_right, None, **self.lk_params
            )
            flow_back_time = (time.time() - start_time) * 1000
            stereo_matching_time += flow_back_time
            
            # Check consistency
            flow_back_mask = np.zeros(len(p0_left), dtype=bool)
            for i in range(len(p0_left)):
                if st_right[i] and st_back[i]:
                    dist = np.linalg.norm(p0_left[i] - p1_back[i])
                    if dist < 1.0:  # Pixel threshold for consistency
                        flow_back_mask[i] = True
        else:
            flow_back_mask = st_right.flatten().astype(bool)
        
        # Collect initial matches after flow consistency
        pts_left = []
        pts_right = []
        valid_indices = []
        
        for i in range(len(kp_left)):
            if flow_back_mask[i] and p1_right[i] is not None:
                pts_left.append(kp_left[i].pt)
                pts_right.append(p1_right[i][0])
                valid_indices.append(i)
        
        # Apply geometric verification
        if self.use_ransac and len(pts_left) >= 8:
            # Use RANSAC to find fundamental matrix
            pts_l = np.array(pts_left, dtype=np.float32)
            pts_r = np.array(pts_right, dtype=np.float32)
            
            F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 3.0, 0.99)
            
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                good_indices = [valid_indices[i] for i in range(len(valid_indices)) if inlier_mask[i]]
            else:
                good_indices = []
        else:
            # Fallback to epipolar constraint
            good_indices = []
            for idx in valid_indices:
                pt_l = kp_left[idx].pt
                pt_r = p1_right[idx][0]
                if abs(pt_r[1] - pt_l[1]) <= self.epipolar_threshold:
                    good_indices.append(idx)
        
        # Create final matches
        good_stereo_matches = []
        kp_right = []
        for idx in good_indices:
            kp_right.append(cv2.KeyPoint(p1_right[idx][0][0], p1_right[idx][0][1], kp_left[idx].size))
            good_stereo_matches.append(cv2.DMatch(idx, len(kp_right)-1, 0))
        
        # Temporal tracking: Track features from previous frame to current
        temporal_matches = []
        if self.prev_left_img is not None and self.trackable_points is not None:
            start_time = time.time()
            p0_prev = np.array(self.trackable_points, dtype=np.float32).reshape(-1, 1, 2)
            p1_curr, st_temp, err_temp = cv2.calcOpticalFlowPyrLK(
                self.prev_left_img, img_left, p0_prev, None, **self.lk_params
            )
            temporal_matching_time = (time.time() - start_time) * 1000
            
            # Update tracked features
            num_lost = self._update_temporal_tracking(p1_curr, st_temp)
            
            # Add exactly the number of features that were lost
            if num_lost > 0:
                num_added = self._add_new_features(img_left, num_lost)
                if num_added > 0:
                    rospy.loginfo_throttle(5.0, f"Replenished {num_added} features (lost {num_lost})")
        else:
            # Initialize trackable points with detected keypoints
            self.trackable_points = [kp.pt for kp in kp_left]
            self.temporal_tracks = [[pt] for pt in self.trackable_points]
            temporal_matching_time = 0.0
        
        # Update previous frame data
        self.prev_left_img = img_left.copy()
        self.prev_left_kp = kp_left
        
        return {
            'keypoints_left': kp_left,
            'keypoints_right': kp_right,
            'stereo_matches': good_stereo_matches,
            'temporal_matches': temporal_matches,
            'num_detected': len(kp_left),
            'num_stereo_matches': len(good_stereo_matches),
                        'num_temporal_tracks': len(self.temporal_tracks),
            'detection_time': detection_time,
            'stereo_matching_time': stereo_matching_time,
            'temporal_matching_time': temporal_matching_time
        }
    
    def _update_temporal_tracking(self, new_points, status):
        """Update temporal feature tracks.
        
        Returns:
            int: Number of features that were lost in this update
        """
        num_before = len(self.trackable_points)
        
        # Update existing tracks with new positions
        new_tracks = []
        new_trackable_points = []
        
        for i, (new_pt, st) in enumerate(zip(new_points, status)):
            if st and i < len(self.temporal_tracks):
                # Extend existing track with new position
                track = self.temporal_tracks[i] + [tuple(new_pt[0])]
                # Keep only last 10 positions to prevent memory issues
                track = track[-10:]
                new_tracks.append(track)
                new_trackable_points.append(tuple(new_pt[0]))
        
        self.temporal_tracks = new_tracks
        self.trackable_points = new_trackable_points
        
        num_lost = num_before - len(self.trackable_points)
        return num_lost
    
    def _add_new_features(self, img_left, num_to_add):
        """Add exactly num_to_add new features to tracking queue.
        
        Args:
            img_left (np.array): Current left image
            num_to_add (int): Number of new features to add
            
        Returns:
            int: Actual number of features added
        """
        if num_to_add <= 0:
            return 0
            
        # Create mask to avoid overlapping with existing tracked features
        mask = np.ones(img_left.shape, dtype=np.uint8) * 255
        
        # Mask out regions around existing tracked features
        for pt in self.trackable_points:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(mask, (x, y), 20, 0, -1)  # 20 pixel radius exclusion zone
        
        # Detect new features in unmasked regions
        new_kp = self.fast.detect(img_left, mask)
        
        # Filter out keypoints that are too close to existing ones
        filtered_new_kp = []
        for kp in new_kp:
            too_close = False
            for existing_pt in self.trackable_points:
                dist = np.linalg.norm(np.array(kp.pt) - np.array(existing_pt))
                if dist < 15:  # Minimum distance threshold
                    too_close = True
                    break
            if not too_close:
                filtered_new_kp.append(kp)
        
        # Add exactly num_to_add features (or as many as available)
        actual_to_add = min(num_to_add, len(filtered_new_kp), self.max_features - len(self.trackable_points))
        
        if actual_to_add > 0:
            # Sort by response (strength) and take the best ones
            filtered_new_kp.sort(key=lambda kp: kp.response, reverse=True)
            new_points = [filtered_new_kp[i].pt for i in range(actual_to_add)]
            new_tracks = [[pt] for pt in new_points]
            
            self.trackable_points.extend(new_points)
            self.temporal_tracks.extend(new_tracks)
        
        return actual_to_add
    
    def _empty_result(self):
        """Return empty result when matching fails."""
        return {
            'keypoints_left': [],
            'keypoints_right': [],
            'stereo_matches': [],
            'temporal_matches': [],
            'num_detected': 0,
            'num_stereo_matches': 0,
            'num_temporal_tracks': 0,
            'detection_time': 0,
            'stereo_matching_time': 0,
            'temporal_matching_time': 0
        }


class OpticalFlowMatchingNode:
    """ROS node for optical flow-based feature matching."""
    
    def __init__(self):
        """Initialize the optical flow matching node."""
        rospy.init_node('optical_flow_matching_node', anonymous=False)

        try:
            self._initialize_from_config()
        except ConfigError as exc:
            rospy.logfatal(f"Configuration error: {exc}")
            raise

        # Initialize matcher
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.matcher = OpticalFlowMatcher(
            self.epipolar_threshold,
            lk_params,
            self.use_ransac,
            self.use_flow_back,
            self.max_features,
            self.min_features
        )
        self.bridge = CvBridge()

        # Publishers
        self.match_viz_pub = rospy.Publisher(
            self.matches_image_topic, Image, queue_size=5
        )
        self.temporal_tracks_viz_pub = rospy.Publisher(
            self.temporal_tracks_image_topic, Image, queue_size=5
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

        rospy.loginfo("Optical Flow Matching Node initialized")
        rospy.loginfo(f"  Epipolar threshold: {self.epipolar_threshold}")
        rospy.loginfo(f"  Max features: {self.max_features}")
        rospy.loginfo(f"  Min features: {self.min_features}")
        rospy.loginfo(f"  Use RANSAC: {self.use_ransac}")
        rospy.loginfo(f"  Use flow back: {self.use_flow_back}")
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
            self.config, self.node_name, expected_type='optical_flow_matching'
        )

        self.left_camera_label = node_block.get('left_camera')
        self.right_camera_label = node_block.get('right_camera')
        if not self.left_camera_label or not self.right_camera_label:
            raise ConfigError(
                f"Node '{self.node_name}' must define 'left_camera' and 'right_camera'"
            )

        matching_cfg = self.config.get('processing', {}).get('matching', {})
        epipolar = matching_cfg.get('epipolar_threshold', 2.0)
        delay = matching_cfg.get('max_delay', 0.1)
        use_ransac = matching_cfg.get('use_ransac', True)
        use_flow_back = matching_cfg.get('use_flow_back', True)
        max_features = matching_cfg.get('max_features', 100)
        min_features = matching_cfg.get('min_features', 80)

        self.epipolar_threshold = float(epipolar)
        self.max_delay = float(delay)
        self.use_ransac = bool(use_ransac)
        self.use_flow_back = bool(use_flow_back)
        self.max_features = int(max_features)
        self.min_features = int(min_features)

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

        outputs_cfg = topics_cfg.get('outputs', {}).get('optical_flow_matching', {})
        self.matches_image_topic = outputs_cfg.get('matches_image')
        self.temporal_tracks_image_topic = outputs_cfg.get('temporal_tracks_image')
        self.match_stats_topic = outputs_cfg.get('match_stats')

        if not self.matches_image_topic or not self.temporal_tracks_image_topic or not self.match_stats_topic:
            raise ConfigError(
                "Optical flow matching output topics are not fully defined"
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
            if self.visualize:
                if self.match_viz_pub.get_num_connections() > 0:
                    self.publish_stereo_visualization(
                        left_msg.header, img_left, img_right, result
                    )
                if self.temporal_tracks_viz_pub.get_num_connections() > 0:
                    self.publish_temporal_visualization(
                        left_msg.header, img_left, result
                    )
            
            total_time = (result['detection_time'] + 
                         result['stereo_matching_time'] + 
                         result['temporal_matching_time'])
            rospy.loginfo_throttle(
                2.0,
                f"Optical Flow: {result['num_stereo_matches']} stereo matches, "
                f"{result['num_temporal_tracks']} temporal tracks "
                f"({total_time:.1f}ms)"
            )
            if result['num_temporal_tracks'] > 0:
                track_lengths = [len(track) for track in self.matcher.temporal_tracks]
                avg_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0
                max_length = max(track_lengths) if track_lengths else 0
                rospy.loginfo_throttle(2.0, f"Track stats: avg={avg_length:.1f}, max={max_length}")
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing stereo pair: {e}")
    
    def publish_stats(self, header, result):
        """Publish matching statistics."""
        stats = MatchStats()
        stats.header = header
        stats.detector_type = 'fast_lk'
        stats.num_initial_matches = result['num_detected']
        stats.num_ratio_filtered = result['num_stereo_matches']  # Using this for stereo matches
        stats.num_epipolar_filtered = result['num_stereo_matches']
        stats.num_final_matches = result['num_stereo_matches']
        stats.detection_time = result['detection_time']
        stats.matching_time = result['stereo_matching_time']
        stats.filtering_time = result['temporal_matching_time']
        stats.total_time = (
            result['detection_time'] +
            result['stereo_matching_time'] +
            result['temporal_matching_time']
        )
        stats.ratio_threshold = 0.0  # Not applicable for optical flow
        stats.epipolar_threshold = self.epipolar_threshold
        
        self.stats_pub.publish(stats)
    
    def publish_stereo_visualization(self, header, img_left, img_right, result):
        """Publish visualization with stereo matched features."""
        # Ensure color images
        left_bgr = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR) if img_left.ndim == 2 else img_left.copy()
        right_bgr = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR) if img_right.ndim == 2 else img_right.copy()

        h = max(left_bgr.shape[0], right_bgr.shape[0])
        w_left = left_bgr.shape[1]
        w_right = right_bgr.shape[1]

        # Create side-by-side canvas
        match_img = np.zeros((h, w_left + w_right, 3), dtype=np.uint8)
        match_img[:left_bgr.shape[0], :w_left] = left_bgr
        match_img[:right_bgr.shape[0], w_left:w_left + w_right] = right_bgr

        # Colors (BGR)
        color_left = (255, 0, 0)   # Blue for left keypoints
        color_right = (0, 255, 0)  # Green for right keypoints
        color_line = (0, 255, 255) # Yellow for connecting lines

        # Draw stereo matches
        for match in result['stereo_matches']:
            kp_l = result['keypoints_left'][match.queryIdx]
            kp_r = result['keypoints_right'][match.trainIdx]
            
            ptL = tuple(np.round(kp_l.pt).astype(int))
            ptR = tuple(np.round(kp_r.pt).astype(int))
            
            cv2.circle(match_img, ptL, 3, color_left, -1, lineType=cv2.LINE_AA)
            cv2.circle(match_img, (ptR[0] + w_left, ptR[1]), 3, color_right, -1, lineType=cv2.LINE_AA)
            cv2.line(match_img, ptL, (ptR[0] + w_left, ptR[1]), color_line, 1, lineType=cv2.LINE_AA)

        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        text1 = "FAST + Lucas-Kanade Optical Flow - Stereo Matches"
        text2 = f"Detected: {result['num_detected']}"
        text3 = f"Stereo Matches: {result['num_stereo_matches']}"
        total_time = result['detection_time'] + result['stereo_matching_time'] + result['temporal_matching_time']
        text4 = f"Time: {total_time:.1f}ms"

        # Draw text with black background
        y_offset = 25
        for i, text in enumerate([text1, text2, text3, text4]):
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
            rospy.logerr(f"Failed to publish stereo visualization: {e}")
    
    def publish_temporal_visualization(self, header, img_left, result):
        """Publish visualization with temporal feature tracks."""
        # Ensure color image
        left_bgr = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR) if img_left.ndim == 2 else img_left.copy()
        
        # Colors (BGR)
        color_current = (0, 255, 255)  # Yellow for current positions
        color_track = (255, 0, 255)    # Magenta for track lines
        color_prev = (255, 255, 0)     # Cyan for previous positions
        
        # Draw temporal tracks
        track_img = left_bgr.copy()
        
        # Draw all temporal tracks
        for track_id, track in enumerate(self.matcher.temporal_tracks):
            if len(track) >= 2:
                # Draw track history
                for i in range(len(track) - 1):
                    pt1 = tuple(np.round(track[i]).astype(int))
                    pt2 = tuple(np.round(track[i + 1]).astype(int))
                    cv2.line(track_img, pt1, pt2, color_track, 2, lineType=cv2.LINE_AA)
                
                # Draw previous positions (smaller circles)
                for i in range(len(track) - 1):
                    pt = tuple(np.round(track[i]).astype(int))
                    cv2.circle(track_img, pt, 2, color_prev, -1, lineType=cv2.LINE_AA)
                
                # Draw current position (larger circle)
                pt_current = tuple(np.round(track[-1]).astype(int))
                cv2.circle(track_img, pt_current, 4, color_current, 2, lineType=cv2.LINE_AA)
                
                # Add track ID
                cv2.putText(track_img, f"{track_id}", pt_current, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_current, 1)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        text1 = "Temporal Feature Tracks"
        text2 = f"Active Tracks: {len(self.matcher.temporal_tracks)}/{self.matcher.max_features}"
        text3 = f"Target Features: {self.matcher.min_features}"
        text4 = f"Temporal Time: {result['temporal_matching_time']:.1f}ms"
        
        # Draw text with black background
        y_offset = 25
        for i, text in enumerate([text1, text2, text3, text4]):
            y = y_offset + i * 30
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(track_img, (5, y - text_h - 5), (15 + text_w, y + 5), (0, 0, 0), -1)
            cv2.putText(track_img, text, (10, y), font, font_scale, (0, 255, 0), thickness)
        
        # Convert to ROS Image
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(track_img, encoding='bgr8')
            viz_msg.header = header
            self.temporal_tracks_viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish temporal visualization: {e}")
    
    def run(self):
        """Keep the node running."""
        rospy.spin()


def main():
    """Main entry point."""
    try:
        node = OpticalFlowMatchingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()