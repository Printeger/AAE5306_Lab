#!/usr/bin/env python3
"""
Feature Detection ROS Node

This node subscribes to stereo camera images and publishes detected features
with visualization overlays and statistics.

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
from aae5306_stereo_vision.msg import FeatureStats


def check_sift_available():
    """Check if SIFT is available in OpenCV."""
    return hasattr(cv2, 'SIFT_create')


class FeatureDetector:
    """Detect features using multiple algorithms."""
    
    def __init__(self, detector_type='sift'):
        """
        Initialize feature detector.
        
        Args:
            detector_type (str): Type of detector ('sift', 'orb', 'fast', 'harris')
        """
        self.detector_type = detector_type.lower()
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the appropriate detector."""
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
        elif self.detector_type == 'fast':
            self.detector = cv2.FastFeatureDetector_create(threshold=10)
        elif self.detector_type == 'harris':
            self.detector = None  # Harris handled separately
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def detect(self, image):
        """
        Detect features in an image.
        
        Args:
            image (np.array): Input image (grayscale)
            
        Returns:
            tuple: (keypoints, descriptors, computation_time)
        """
        start_time = time.time()
        
        if self.detector_type == 'harris':
            keypoints, descriptors = self._detect_harris(image)
        elif self.detector_type == 'fast':
            keypoints = self.detector.detect(image, None)
            descriptors = None
        else:
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        computation_time = (time.time() - start_time) * 1000  # ms
        
        return keypoints, descriptors, computation_time
    
    def _detect_harris(self, image):
        """Detect Harris corners."""
        blockSize = 2
        ksize = 3
        k = 0.04
        
        dst = cv2.cornerHarris(image, blockSize, ksize, k)
        dst = cv2.dilate(dst, None)
        threshold = 0.01 * dst.max()
        corner_locations = np.where(dst > threshold)
        
        keypoints = []
        for y, x in zip(corner_locations[0], corner_locations[1]):
            kp = cv2.KeyPoint(float(x), float(y), 1)
            keypoints.append(kp)
        
        return keypoints, None


class FeatureDetectionNode:
    """ROS node for real-time feature detection."""
    
    def __init__(self):
        """Initialize the feature detection node."""
        rospy.init_node('feature_detection_node', anonymous=False)
        
        # Parameters
        self.detector_type = rospy.get_param('~detector_type', 'sift')
        self.input_topic = rospy.get_param('~input_topic', '/cam0/image_raw')
        self.visualize = rospy.get_param('~visualize', True)
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)
        
        # Initialize detector
        self.detector = FeatureDetector(self.detector_type)
        self.bridge = CvBridge()
        
        # Publishers
        self.feature_viz_pub = rospy.Publisher(
            '~features_image', Image, queue_size=5
        )
        self.stats_pub = rospy.Publisher(
            '~feature_stats', FeatureStats, queue_size=10
        )
        
        # Subscriber
        self.image_sub = rospy.Subscriber(
            self.input_topic, Image, self.image_callback, queue_size=5
        )
        
        # State
        self.last_publish_time = rospy.Time.now()
        self.min_publish_interval = rospy.Duration(1.0 / self.publish_rate)
        
        rospy.loginfo(f"Feature Detection Node initialized")
        rospy.loginfo(f"  Detector type: {self.detector_type}")
        rospy.loginfo(f"  Input topic: {self.input_topic}")
        rospy.loginfo(f"  Visualize: {self.visualize}")
        rospy.loginfo(f"  Publish rate: {self.publish_rate} Hz")
    
    def image_callback(self, msg):
        """Process incoming image and detect features."""
        # Rate limiting
        current_time = rospy.Time.now()
        if (current_time - self.last_publish_time) < self.min_publish_interval:
            return
        self.last_publish_time = current_time
        
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            
            # Detect features
            keypoints, descriptors, comp_time = self.detector.detect(cv_image)
            
            # Publish statistics
            self.publish_stats(msg.header, keypoints, comp_time, cv_image.shape)
            
            # Publish visualization
            if self.visualize and self.feature_viz_pub.get_num_connections() > 0:
                self.publish_visualization(msg.header, cv_image, keypoints)
            
            rospy.loginfo_throttle(
                2.0, 
                f"Detected {len(keypoints)} features in {comp_time:.1f}ms"
            )
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def publish_stats(self, header, keypoints, comp_time, image_shape):
        """Publish feature statistics."""
        stats = FeatureStats()
        stats.header = header
        stats.detector_type = self.detector_type
        stats.num_keypoints = len(keypoints)
        stats.computation_time = comp_time
        stats.image_height = image_shape[0]
        stats.image_width = image_shape[1]
        
        self.stats_pub.publish(stats)
    
    def publish_visualization(self, header, image, keypoints):
        """Publish visualization with drawn keypoints."""
        # Convert grayscale to BGR for colored keypoints
        if len(image.shape) == 2:
            viz_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            viz_image = image.copy()
        
        # Draw keypoints
        viz_image = cv2.drawKeypoints(
            viz_image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        
        # Add text overlays with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Title
        text1 = f"Detector: {self.detector_type.upper()}"
        text2 = f"Features: {len(keypoints)}"
        text3 = f"Topic: {self.input_topic}"
        
        # Draw text with black background for better visibility
        y_offset = 25
        for i, text in enumerate([text1, text2, text3]):
            y = y_offset + i * 30
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(viz_image, (5, y - text_h - 5), (15 + text_w, y + 5), (0, 0, 0), -1)
            cv2.putText(viz_image, text, (10, y), font, font_scale, (0, 255, 0), thickness)
        
        # Convert back to ROS Image
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
            viz_msg.header = header
            self.feature_viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish visualization: {e}")
    
    def run(self):
        """Keep the node running."""
        rospy.spin()


def main():
    """Main entry point."""
    try:
        node = FeatureDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()
