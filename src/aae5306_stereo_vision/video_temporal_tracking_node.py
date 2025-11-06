#!/usr/bin/env python3
"""
Optical Flow Video Tracking ROS Node

This node reads frames from an MP4 video file, extracts features on a moving
object, and tracks them over time using Lucas-Kanade optical flow.

Author: AAE5306 Teaching Team
Date: November 6, 2025
"""

import time

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from aae5306_stereo_vision.msg import MatchStats

# Minimal inline config loader to keep the project simple
import threading
from typing import Any, Dict, List, Optional, Tuple

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


class OpticalFlowVideoTracker:

    def __init__(
        self,
        max_features=150,
        min_features=80,
        motion_threshold=1.0,
        quality_level=0.01,
        min_distance=7,
        fb_threshold=1.5,
        err_threshold=20.0,
        track_length=30,
        bg_history=300,
        bg_var_threshold=32.0,
    ):
        self.max_features = int(max_features)
        self.min_features = int(min_features)
        self.motion_threshold = float(motion_threshold)
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.fb_threshold = float(fb_threshold)
        self.err_threshold = float(err_threshold)
        self.track_length = int(track_length)
        self.bg_history = int(bg_history)
        self.bg_var_threshold = float(bg_var_threshold)

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )

        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.reset()

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.tracks = []
        self.next_track_id = 0
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history,
            varThreshold=self.bg_var_threshold,
            detectShadows=False
        )

    def process_frame(self, gray):
        detection_time = 0.0
        matching_time = 0.0
        filtering_time = 0.0
        num_candidates = 0
        num_new_features = 0

        fg_mask, mask_time = self._foreground_mask(gray)
        filtering_time += mask_time

        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            added, add_time = self._add_new_features(gray, fg_mask, allow_full_frame=True)
            detection_time += add_time
            num_new_features += added
            self.prev_gray = gray.copy()
            return {
                'tracks': self._snapshot_tracks(),
                'detection_time': detection_time,
                'matching_time': matching_time,
                'filtering_time': filtering_time,
                'num_candidates': num_candidates,
                'num_active_tracks': len(self.tracks),
                'num_new_features': num_new_features,
            }

        prev_points = self.prev_points
        num_candidates = len(prev_points) if prev_points is not None else 0

        start = time.perf_counter()
        next_points, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_points, None, **self.lk_params
        )
        matching_time += (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        reproj_points, st_back, err_back = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, next_points, None, **self.lk_params
        )
        matching_time += (time.perf_counter() - start) * 1000.0

        fb_error = np.linalg.norm(prev_points - reproj_points, axis=2).reshape(-1)
        forward_ok = st.reshape(-1).astype(bool)
        backward_ok = st_back.reshape(-1).astype(bool)
        error_ok = err.reshape(-1) < self.err_threshold
        fb_ok = fb_error < self.fb_threshold
        displacement = np.linalg.norm(next_points - prev_points, axis=2).reshape(-1)
        mask_ok = self._mask_filter(next_points, fg_mask)

        start = time.perf_counter()
        updated_tracks = []
        new_points = []
        for idx, track in enumerate(self.tracks):
            if idx >= len(next_points):
                break
            if not (forward_ok[idx] and backward_ok[idx] and error_ok[idx] and fb_ok[idx]):
                continue

            pt = next_points[idx][0]
            if not mask_ok[idx] and len(track['positions']) <= 1:
                continue

            if displacement[idx] < self.motion_threshold and len(track['positions']) <= 1:
                continue

            track['positions'].append((float(pt[0]), float(pt[1])))
            if len(track['positions']) > self.track_length:
                track['positions'] = track['positions'][-self.track_length:]

            updated_tracks.append(track)
            new_points.append([[float(pt[0]), float(pt[1])]])

        filtering_time += (time.perf_counter() - start) * 1000.0

        if new_points:
            self.prev_points = np.asarray(new_points, dtype=np.float32)
        else:
            self.prev_points = None

        self.tracks = updated_tracks

        if self.prev_points is None or len(self.prev_points) < self.min_features:
            added, add_time = self._add_new_features(gray, fg_mask)
            detection_time += add_time
            num_new_features += added

        self.prev_gray = gray.copy()

        return {
            'tracks': self._snapshot_tracks(),
            'detection_time': detection_time,
            'matching_time': matching_time,
            'filtering_time': filtering_time,
            'num_candidates': num_candidates,
            'num_active_tracks': len(self.tracks),
            'num_new_features': num_new_features,
        }

    def _foreground_mask(self, gray):
        start = time.perf_counter()
        mask = self.background_subtractor.apply(gray)
        if mask is None:
            mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            mask = cv2.medianBlur(mask, 5)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=1)
        elapsed = (time.perf_counter() - start) * 1000.0
        return mask, elapsed

    def _mask_filter(self, points, mask):
        if mask is None or mask.size == 0:
            return np.ones((len(points),), dtype=bool)
        h, w = mask.shape
        flags = []
        for pt in points.reshape(-1, 2):
            x = int(np.clip(pt[0], 0, w - 1))
            y = int(np.clip(pt[1], 0, h - 1))
            flags.append(mask[y, x] > 0)
        return np.asarray(flags, dtype=bool)

    def _add_new_features(self, gray, fg_mask, allow_full_frame=False):
        start = time.perf_counter()
        candidate_mask = None
        if fg_mask is not None:
            nz = cv2.countNonZero(fg_mask)
            if nz > 0:
                candidate_mask = cv2.medianBlur(fg_mask, 5)
                _, candidate_mask = cv2.threshold(candidate_mask, 1, 255, cv2.THRESH_BINARY)
        if candidate_mask is None and allow_full_frame:
            candidate_mask = np.full_like(gray, 255, dtype=np.uint8)
        if candidate_mask is not None:
            candidate_mask = self._suppress_existing_points(candidate_mask)

        max_corners = max(self.max_features - len(self.tracks), 0)
        num_added = 0
        pts = None
        if max_corners > 0 and candidate_mask is not None:
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_corners,
                mask=candidate_mask,
                **self.feature_params
            )
        elif max_corners > 0 and allow_full_frame:
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_corners,
                mask=None,
                **self.feature_params
            )

        if pts is not None:
            pts = np.asarray(pts, dtype=np.float32)
            num_added = int(pts.shape[0])
            for pt in pts.reshape(-1, 2):
                track = {
                    'id': self.next_track_id,
                    'positions': [(float(pt[0]), float(pt[1]))]
                }
                self.next_track_id += 1
                self.tracks.append(track)
            if self.prev_points is None or len(self.prev_points) == 0:
                self.prev_points = pts.reshape(-1, 1, 2)
            else:
                self.prev_points = np.concatenate(
                    (self.prev_points, pts.reshape(-1, 1, 2)), axis=0
                )

        elapsed = (time.perf_counter() - start) * 1000.0
        return num_added, elapsed

    def _suppress_existing_points(self, mask):
        if mask is None:
            return None
        suppressed = mask.copy()
        if self.prev_points is not None and len(self.prev_points) > 0:
            for pt in self.prev_points.reshape(-1, 2):
                cv2.circle(
                    suppressed,
                    (int(round(pt[0])), int(round(pt[1]))),
                    self.min_distance,
                    0,
                    -1
                )
        return suppressed

    def _snapshot_tracks(self):
        snapshot = []
        for track in self.tracks:
            snapshot.append({
                'id': track['id'],
                'positions': list(track['positions'])
            })
        return snapshot


class VideoOpticalFlowTrackingNode:
    def __init__(self):
        rospy.init_node('video_temporal_tracking_node', anonymous=False)

        try:
            self._initialize_from_config()
        except ConfigError as exc:
            rospy.logfatal(f"Configuration error: {exc}")
            raise

        tracking_cfg = self.config.get('processing', {}).get('tracking', {})
        quality_level = float(tracking_cfg.get('quality_level', 0.01))
        min_distance = int(tracking_cfg.get('min_distance', 7))
        fb_threshold = float(tracking_cfg.get('fb_threshold', 1.5))
        err_threshold = float(tracking_cfg.get('error_threshold', 20.0))
        track_history = int(tracking_cfg.get('track_history', 30))
        bg_history = int(tracking_cfg.get('bg_history', 300))
        bg_var_threshold = float(tracking_cfg.get('bg_var_threshold', 32.0))

        self.tracker = OpticalFlowVideoTracker(
            max_features=self.max_features,
            min_features=self.min_features,
            motion_threshold=self.motion_threshold,
            quality_level=quality_level,
            min_distance=min_distance,
            fb_threshold=fb_threshold,
            err_threshold=err_threshold,
            track_length=track_history,
            bg_history=bg_history,
            bg_var_threshold=bg_var_threshold
        )

        self.bridge = CvBridge()

        self.frame_pub = rospy.Publisher(
            self.frame_image_topic, Image, queue_size=5
        )
        self.temporal_tracks_viz_pub = rospy.Publisher(
            self.temporal_tracks_image_topic, Image, queue_size=5
        )
        self.stats_pub = rospy.Publisher(
            self.track_stats_topic, MatchStats, queue_size=10
        )

        self.cap = None
        self.timer = rospy.Timer(
            rospy.Duration(max(1.0 / max(self.frame_rate, 1e-3), 1e-3)),
            self.process_video_frame
        )

        rospy.loginfo("Optical Flow Video Tracking Node initialized")
        rospy.loginfo(f"  Video file: {self.video_file}")
        rospy.loginfo(f"  Frame rate: {self.frame_rate} Hz")
        rospy.loginfo(f"  Loop video: {self.loop_video}")
        rospy.loginfo(f"  Visualize: {self.visualize}")
        rospy.loginfo(
            "  Tracking params: max=%d min=%d motion>=%.2f px",
            self.max_features,
            self.min_features,
            self.motion_threshold
        )

    def _initialize_from_config(self):
        self.node_name = rospy.get_name().split('/')[-1]
        self.config = get_pipeline_config()
        self.config_version = self.config.get('config_version')

        self.node_block = get_node_block(
            self.config, self.node_name, expected_type='video_temporal_tracking'
        )

        self.video_file = self.node_block.get('video_file')
        if not self.video_file:
            raise ConfigError("Video file path must be specified")

        self.frame_rate = float(self.node_block.get('frame_rate', 30.0))
        self.frame_rate = float(rospy.get_param('~frame_rate', self.frame_rate))
        self.video_file = rospy.get_param('~video_file', self.video_file)
        self.visualize = bool(self.node_block.get('visualize', True))
        self.visualize = bool(rospy.get_param('~visualize', self.visualize))
        self.loop_video = bool(self.node_block.get('loop_video', True))
        self.loop_video = bool(rospy.get_param('~loop_video', self.loop_video))

        tracking_cfg = self.config.get('processing', {}).get('tracking', {})
        self.max_features = int(tracking_cfg.get('max_features', 150))
        self.min_features = int(tracking_cfg.get('min_features', 80))
        self.motion_threshold = float(tracking_cfg.get('motion_threshold', 1.0))

        topics_cfg = self.config.get('topics', {}).get('outputs', {}).get('video_temporal_tracking', {})
        self.frame_image_topic = topics_cfg.get('frame_image')
        self.temporal_tracks_image_topic = topics_cfg.get('temporal_tracks_image')
        self.track_stats_topic = topics_cfg.get('track_stats')

        if not self.frame_image_topic or not self.temporal_tracks_image_topic or not self.track_stats_topic:
            raise ConfigError("Video temporal tracking output topics are not fully defined")

    def _ensure_capture(self):
        if self.cap is not None and self.cap.isOpened():
            return True
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            rospy.logerr_throttle(5.0, f"Failed to open video file: {self.video_file}")
            self.cap = None
            return False
        self.tracker.reset()
        rospy.loginfo(f"Started reading video: {self.video_file}")
        return True

    def process_video_frame(self, event):
        if not self._ensure_capture():
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.loop_video:
                rospy.loginfo_throttle(5.0, "Reached end of video, rewinding")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.tracker.reset()
            else:
                rospy.loginfo_once("Video processing completed")
                self.cap.release()
                self.cap = None
            return

        if frame.ndim == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        try:
            result = self.tracker.process_frame(gray_frame)
        except Exception as exc:
            rospy.logerr(f"Tracker failure: {exc}")
            return

        tracks = result['tracks']

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "video_frame"

        self.publish_stats(header, result)

        if self.visualize and self.frame_pub.get_num_connections() > 0:
            self.publish_frame(header, frame)

        if self.visualize and self.temporal_tracks_viz_pub.get_num_connections() > 0:
            self.publish_motion_visualization(header, frame, tracks, result)

        if tracks:
            avg_length = sum(len(track['positions']) for track in tracks) / max(len(tracks), 1)
            rospy.loginfo_throttle(
                2.0,
                "Active tracks: %d | Avg length: %.1f | Added: %d",
                len(tracks),
                avg_length,
                result['num_new_features']
            )

    def publish_stats(self, header, result):
        stats = MatchStats()
        stats.header = header
        stats.detector_type = 'lk_optical_flow'
        stats.num_initial_matches = int(result['num_candidates'])
        stats.num_ratio_filtered = int(result['num_active_tracks'])
        stats.num_epipolar_filtered = int(result['num_active_tracks'])
        stats.num_final_matches = int(result['num_active_tracks'])
        stats.detection_time = float(result['detection_time'])
        stats.matching_time = float(result['matching_time'])
        stats.filtering_time = float(result['filtering_time'])
        stats.total_time = stats.detection_time + stats.matching_time + stats.filtering_time
        stats.ratio_threshold = 0.0
        stats.epipolar_threshold = 0.0
        self.stats_pub.publish(stats)

    def publish_frame(self, header, frame):
        try:
            frame_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            frame_msg.header = header
            self.frame_pub.publish(frame_msg)
        except CvBridgeError:
            rospy.logwarn_throttle(5.0, "Failed to publish frame image")

    def publish_motion_visualization(self, header, frame, tracks, result):
        if frame.ndim == 2:
            viz = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            viz = frame.copy()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
            (128, 255, 0)
        ]

        for track in tracks:
            track_id = track['id']
            positions = track['positions']
            color = colors[track_id % len(colors)]

            if len(positions) >= 2:
                for idx in range(len(positions) - 1):
                    pt1 = tuple(map(int, map(round, positions[idx])))
                    pt2 = tuple(map(int, map(round, positions[idx + 1])))
                    cv2.line(viz, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

            if positions:
                pt_curr = tuple(map(int, map(round, positions[-1])))
                cv2.circle(viz, pt_curr, 4, color, -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    viz,
                    f"{track_id}",
                    (pt_curr[0] + 4, pt_curr[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    lineType=cv2.LINE_AA
                )

        overlay_lines = [
            "Optical Flow Video Tracking",
            f"Tracks: {len(tracks)}",
            f"Candidates: {result['num_candidates']}",
            f"New features: {result['num_new_features']}",
            f"Timing ms - detect {result['detection_time']:.1f} | track {result['matching_time']:.1f} | filter {result['filtering_time']:.1f}"
        ]

        y = 25
        for text in overlay_lines:
            cv2.putText(
                viz,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA
            )
            y += 25

        try:
            msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
            msg.header = header
            self.temporal_tracks_viz_pub.publish(msg)
        except CvBridgeError:
            rospy.logwarn_throttle(5.0, "Failed to publish tracks visualization")

    def run(self):
        rospy.spin()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


def main():
    """Main entry point."""
    try:
        node = VideoOpticalFlowTrackingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()

