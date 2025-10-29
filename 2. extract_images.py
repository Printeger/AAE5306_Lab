#!/usr/bin/env python3
"""
Extract synchronized stereo image pairs from EuRoC MAV ROS bag files.

This script reads ROS bag files containing stereo camera data and extracts
synchronized image pairs at specified intervals.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from utils import load_config, ensure_dir, print_config_summary


class StereoImageExtractor:
    """Extract synchronized stereo images from ROS bag."""
    
    def __init__(self, bag_file, output_dir, skip_frames=10, max_pairs=300):
        """
        Initialize the extractor.
        
        Args:
            bag_file (str): Path to ROS bag file
            output_dir (str): Output directory for images
            skip_frames (int): Extract every Nth frame
            max_pairs (int): Maximum number of pairs to extract
        """
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.skip_frames = skip_frames
        self.max_pairs = max_pairs
        self.bridge = CvBridge()
        
        # Create output directories
        self.cam0_dir = os.path.join(output_dir, 'cam0')
        self.cam1_dir = os.path.join(output_dir, 'cam1')
        os.makedirs(self.cam0_dir, exist_ok=True)
        os.makedirs(self.cam1_dir, exist_ok=True)
        
        # Storage for timestamps
        self.timestamps = []
        
    def get_bag_info(self):
        """Get information about the ROS bag."""
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
            info = bag.get_type_and_topic_info()
            
            print("\n" + "="*60)
            print(f"ROS Bag Information: {os.path.basename(self.bag_file)}")
            print("="*60)
            
            # Print topics
            print("\nAvailable topics:")
            for topic, topic_info in info.topics.items():
                print(f"  {topic}")
                print(f"    Type: {topic_info.msg_type}")
                print(f"    Messages: {topic_info.message_count}")
                print(f"    Frequency: {topic_info.frequency:.2f} Hz")
            
            # Get duration
            start_time = bag.get_start_time()
            end_time = bag.get_end_time()
            duration = end_time - start_time
            
            print(f"\nDuration: {duration:.2f} seconds")
            print(f"Start time: {start_time:.2f}")
            print(f"End time: {end_time:.2f}")
            print("="*60 + "\n")
            
            bag.close()
            return info
            
        except Exception as e:
            print(f"Error reading bag file: {e}")
            sys.exit(1)
    
    def extract_images(self, cam0_topic='/cam0/image_raw', cam1_topic='/cam1/image_raw'):
        """
        Extract synchronized stereo image pairs.
        
        Args:
            cam0_topic (str): Topic name for left camera
            cam1_topic (str): Topic name for right camera
        """
        print(f"Extracting images from:")
        print(f"  Left camera:  {cam0_topic}")
        print(f"  Right camera: {cam1_topic}")
        print(f"  Skip frames:  {self.skip_frames}")
        print(f"  Max pairs:    {self.max_pairs}\n")
        
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
            
            # Get total message count for progress bar
            total_messages = bag.get_message_count(topic_filters=[cam0_topic, cam1_topic])
            
            # Storage for buffering messages
            cam0_buffer = {}
            cam1_buffer = {}
            
            pair_count = 0
            frame_count = 0
            
            # Read messages
            with tqdm(total=total_messages, desc="Processing messages") as pbar:
                for topic, msg, t in bag.read_messages(topics=[cam0_topic, cam1_topic]):
                    timestamp = msg.header.stamp.to_sec()
                    
                    # Store in appropriate buffer
                    if topic == cam0_topic:
                        cam0_buffer[timestamp] = msg
                    elif topic == cam1_topic:
                        cam1_buffer[timestamp] = msg
                    
                    # Try to find synchronized pairs
                    self._process_buffers(cam0_buffer, cam1_buffer, frame_count, pair_count)
                    
                    pbar.update(1)
                    
                    # Check if we have enough pairs
                    if pair_count >= self.max_pairs:
                        break
            
            bag.close()
            
            # Final processing of remaining buffers
            pair_count = self._process_buffers(cam0_buffer, cam1_buffer, frame_count, pair_count, final=True)
            
            print(f"\n✓ Extraction completed!")
            print(f"  Total pairs extracted: {pair_count}")
            print(f"  Output directory: {self.output_dir}")
            
            # Save timestamps
            self._save_timestamps()
            
            return pair_count
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            sys.exit(1)
    
    def _process_buffers(self, cam0_buffer, cam1_buffer, frame_count, pair_count, final=False):
        """
        Process buffered messages to find synchronized pairs.
        
        Args:
            cam0_buffer (dict): Buffer for cam0 messages
            cam1_buffer (dict): Buffer for cam1 messages
            frame_count (int): Current frame count
            pair_count (int): Current pair count
            final (bool): Whether this is final processing
            
        Returns:
            int: Updated pair count
        """
        if not cam0_buffer or not cam1_buffer:
            return pair_count
        
        # Find synchronized pairs (within 0.01 seconds)
        max_time_diff = 0.01
        
        cam0_times = sorted(cam0_buffer.keys())
        cam1_times = sorted(cam1_buffer.keys())
        
        matched_pairs = []
        
        for t0 in cam0_times:
            # Find closest cam1 timestamp
            closest_t1 = min(cam1_times, key=lambda t1: abs(t1 - t0))
            
            if abs(closest_t1 - t0) < max_time_diff:
                matched_pairs.append((t0, closest_t1))
        
        # Extract matched pairs
        for t0, t1 in matched_pairs:
            if frame_count % self.skip_frames == 0:
                if pair_count < self.max_pairs:
                    self._save_image_pair(
                        cam0_buffer[t0],
                        cam1_buffer[t1],
                        pair_count,
                        (t0 + t1) / 2  # Average timestamp
                    )
                    pair_count += 1
            
            frame_count += 1
            
            # Remove processed messages from buffers
            del cam0_buffer[t0]
            del cam1_buffer[t1]
        
        # Clean old messages from buffers (keep only recent ones)
        if not final:
            current_time = max(max(cam0_buffer.keys(), default=0), 
                             max(cam1_buffer.keys(), default=0))
            
            # Keep only messages within 1 second
            cam0_buffer = {t: msg for t, msg in cam0_buffer.items() 
                          if current_time - t < 1.0}
            cam1_buffer = {t: msg for t, msg in cam1_buffer.items() 
                          if current_time - t < 1.0}
        
        return pair_count
    
    def _save_image_pair(self, cam0_msg, cam1_msg, index, timestamp):
        """
        Save a synchronized image pair.
        
        Args:
            cam0_msg: ROS message from cam0
            cam1_msg: ROS message from cam1
            index (int): Pair index
            timestamp (float): Average timestamp
        """
        try:
            # Convert ROS images to OpenCV
            cam0_image = self.bridge.imgmsg_to_cv2(cam0_msg, desired_encoding='mono8')
            cam1_image = self.bridge.imgmsg_to_cv2(cam1_msg, desired_encoding='mono8')
            
            # Generate filenames
            filename = f"{index:06d}.png"
            cam0_path = os.path.join(self.cam0_dir, filename)
            cam1_path = os.path.join(self.cam1_dir, filename)
            
            # Save images
            cv2.imwrite(cam0_path, cam0_image)
            cv2.imwrite(cam1_path, cam1_image)
            
            # Store timestamp
            self.timestamps.append((index, timestamp))
            
        except Exception as e:
            print(f"Warning: Failed to save pair {index}: {e}")
    
    def _save_timestamps(self):
        """Save timestamps to file."""
        timestamp_file = os.path.join(self.output_dir, 'timestamps.txt')
        
        with open(timestamp_file, 'w') as f:
            f.write("# Image pair timestamps\n")
            f.write("# Format: index timestamp\n")
            for index, timestamp in self.timestamps:
                f.write(f"{index:06d} {timestamp:.9f}\n")
        
        print(f"  Timestamps saved to: {timestamp_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract stereo image pairs from EuRoC MAV ROS bag'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get parameters from config
    ie_cfg = config['image_extraction']
    bag_file = config['paths']['rosbag_file']
    output_dir = config['paths']['images_output']
    
    # Check if task is enabled
    if not ie_cfg.get('enabled', True):
        print("Image extraction is disabled in config.yaml")
        return
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Check if bag file exists
    if not os.path.exists(bag_file):
        print(f"Error: Bag file not found: {bag_file}")
        print("Please download the EuRoC dataset ROS bag file first.")
        sys.exit(1)
    
    # Create extractor
    extractor = StereoImageExtractor(
        bag_file=bag_file,
        output_dir=output_dir,
        skip_frames=ie_cfg['skip_frames'],
        max_pairs=ie_cfg['max_pairs']
    )
    
    # Get bag info
    extractor.get_bag_info()
    
    if ie_cfg.get('info_only', False):
        print("Info only mode - extraction skipped.")
        return
    
    # Extract images
    extractor.extract_images(
        cam0_topic=ie_cfg['cam0_topic'],
        cam1_topic=ie_cfg['cam1_topic']
    )
    
    print("\n✓ Image extraction completed successfully!")
    print(f"Extracted {extractor.skip_frames} images to: {output_dir}\n")


if __name__ == '__main__':
    main()