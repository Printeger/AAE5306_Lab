#!/usr/bin/env python3
"""
Feature matching between stereo image pairs using SIFT and ORB.

This script performs robust feature matching with ratio test and
epipolar constraint filtering.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import json
import pickle
import matplotlib.pyplot as plt
from utils import load_config, ensure_dir, get_image_pair_path, print_config_summary


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
        self.matches = None
        self.good_matches = None
        self.filtered_matches = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize detector and matcher."""
        if self.detector_type == 'sift':
            self.detector = cv2.SIFT_create()
            # FLANN matcher for SIFT (L2 distance)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        elif self.detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
            # FLANN matcher for ORB (Hamming distance)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                               table_number=6,
                               key_size=12,
                               multi_probe_level=1)
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
            dict: Matching results with statistics
        """
        print(f"\n{'='*60}")
        print(f"Feature Matching with {self.detector_type.upper()}")
        print(f"{'='*60}\n")
        
        # Detect features
        print("Step 1: Detecting features...")
        start_time = time.time()
        
        kp_left, desc_left = self.detector.detectAndCompute(img_left, None)
        kp_right, desc_right = self.detector.detectAndCompute(img_right, None)
        
        detection_time = (time.time() - start_time) * 1000
        
        print(f"  Left image:  {len(kp_left)} keypoints")
        print(f"  Right image: {len(kp_right)} keypoints")
        print(f"  Detection time: {detection_time:.2f} ms\n")
        
        if desc_left is None or desc_right is None:
            print("Error: No descriptors computed")
            return None
        
        # Match features
        print("Step 2: Matching features...")
        start_time = time.time()
        
        # Find 2 nearest neighbors for ratio test
        matches = self.matcher.knnMatch(desc_left, desc_right, k=2)
        
        matching_time = (time.time() - start_time) * 1000
        print(f"  Initial matches: {len(matches)}")
        print(f"  Matching time: {matching_time:.2f} ms\n")
        
        # Apply ratio test (Lowe's ratio test)
        print(f"Step 3: Applying ratio test (threshold={self.ratio_threshold})...")
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        print(f"  Matches after ratio test: {len(good_matches)}\n")
        
        # Apply epipolar constraint (rectified stereo)
        print(f"Step 4: Applying epipolar constraint (threshold={self.epipolar_threshold} px)...")
        filtered_matches = []
        
        for match in good_matches:
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            
            # For rectified stereo, y-coordinates should be approximately equal
            y_diff = abs(pt_left[1] - pt_right[1])
            
            # Also check that right point is to the left (positive disparity)
            x_diff = pt_left[0] - pt_right[0]
            
            if y_diff < self.epipolar_threshold and x_diff > 0:
                filtered_matches.append(match)
        
        print(f"  Matches after epipolar constraint: {len(filtered_matches)}\n")
        
        # Store results
        self.matches = matches
        self.good_matches = good_matches
        self.filtered_matches = filtered_matches
        
        # Compute statistics
        results = {
            'detector_type': self.detector_type,
            'num_keypoints_left': len(kp_left),
            'num_keypoints_right': len(kp_right),
            'num_initial_matches': len(matches),
            'num_ratio_filtered': len(good_matches),
            'num_final_matches': len(filtered_matches),
            'detection_time_ms': detection_time,
            'matching_time_ms': matching_time,
            'total_time_ms': detection_time + matching_time,
            'ratio_threshold': self.ratio_threshold,
            'epipolar_threshold': self.epipolar_threshold,
            'match_ratio': len(filtered_matches) / min(len(kp_left), len(kp_right)) * 100
        }
        
        # Compute disparity statistics
        disparities = []
        for match in filtered_matches:
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            disparity = pt_left[0] - pt_right[0]
            disparities.append(disparity)
        
        if disparities:
            results['disparity_mean'] = np.mean(disparities)
            results['disparity_std'] = np.std(disparities)
            results['disparity_min'] = np.min(disparities)
            results['disparity_max'] = np.max(disparities)
        
        return results, kp_left, kp_right, filtered_matches
    
    def visualize_matches(self, img_left, img_right, kp_left, kp_right, 
                         matches, output_path=None, max_matches=100):
        """
        Visualize feature matches.
        
        Args:
            img_left (np.array): Left image
            img_right (np.array): Right image
            kp_left (list): Keypoints in left image
            kp_right (list): Keypoints in right image
            matches (list): List of matches
            output_path (str): Path to save visualization
            max_matches (int): Maximum number of matches to display
        """
        # Select subset of matches for visualization
        if len(matches) > max_matches:
            # Sort by distance and take best matches
            matches_sorted = sorted(matches, key=lambda x: x.distance)
            matches_to_draw = matches_sorted[:max_matches]
        else:
            matches_to_draw = matches
        
        # Draw matches
        img_matches = cv2.drawMatches(
            img_left, kp_left,
            img_right, kp_right,
            matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Create figure
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'{self.detector_type.upper()} Matches: {len(matches)} total '
                 f'({len(matches_to_draw)} displayed)',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add statistics
        stats_text = f'Total Matches: {len(matches)}\n'
        stats_text += f'Displayed: {len(matches_to_draw)}\n'
        stats_text += f'Detector: {self.detector_type.upper()}'
        
        plt.text(10, 30, stats_text, fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Match visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_matches(self, kp_left, kp_right, matches, output_file):
        """
        Save matches to file for depth recovery.
        
        Args:
            kp_left (list): Left keypoints
            kp_right (list): Right keypoints
            matches (list): Match list
            output_file (str): Output pickle file path
        """
        # Extract matched point coordinates
        pts_left = []
        pts_right = []
        
        for match in matches:
            pts_left.append(kp_left[match.queryIdx].pt)
            pts_right.append(kp_right[match.trainIdx].pt)
        
        pts_left = np.array(pts_left, dtype=np.float32)
        pts_right = np.array(pts_right, dtype=np.float32)
        
        # Save to pickle file
        match_data = {
            'pts_left': pts_left,
            'pts_right': pts_right,
            'num_matches': len(matches),
            'detector_type': self.detector_type
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(match_data, f)
        
        print(f"Match data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Feature matching between stereo images'
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
    fm_cfg = config['feature_matching']
    images_dir = config['paths']['images_output']
    output_dir = config['paths']['matches_output']
    
    # Check if task is enabled
    if not fm_cfg.get('enabled', True):
        print("Feature matching is disabled in config.yaml")
        return
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Get image paths
    img_index = fm_cfg['test_image_index']
    left_image_path = get_image_pair_path(images_dir, img_index, 'cam0')
    right_image_path = get_image_pair_path(images_dir, img_index, 'cam1')
    
    # Check if images exist
    if not os.path.exists(left_image_path):
        print(f"Error: Left image not found: {left_image_path}")
        print(f"Please run image extraction first (Task 2)")
        sys.exit(1)
    
    if not os.path.exists(right_image_path):
        print(f"Error: Right image not found: {right_image_path}")
        sys.exit(1)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Load images
    print(f"Loading images...")
    print(f"  Image index: {img_index}")
    print(f"  Left:  {left_image_path}")
    print(f"  Right: {right_image_path}")
    
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        print("Error: Failed to load images")
        sys.exit(1)
    
    # Process each detector
    for detector_type in fm_cfg['detectors']:
        print(f"\n{'='*70}")
        print(f"Processing with {detector_type.upper()} detector")
        print(f"{'='*70}")
        
        # Get detector-specific parameters
        detector_params = fm_cfg.get(detector_type, {})
        ratio_threshold = detector_params.get('ratio_threshold', 0.75)
        epipolar_threshold = detector_params.get('epipolar_threshold', 2.0)
        
        # Create matcher
        matcher = FeatureMatcher(
            detector_type=detector_type,
            ratio_threshold=ratio_threshold,
            epipolar_threshold=epipolar_threshold
        )
        
        # Perform matching
        results, kp_left, kp_right, matches = matcher.detect_and_match(img_left, img_right)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Matching Summary")
        print(f"{'='*60}")
        print(f"Detector: {results['detector_type'].upper()}")
        print(f"Initial matches: {results['num_initial_matches']}")
        print(f"After ratio test: {results['num_ratio_filtered']}")
        print(f"Final matches: {results['num_final_matches']}")
        print(f"Match ratio: {results['match_ratio']:.2f}%")
        print(f"Total time: {results['total_time_ms']:.2f} ms")
        
        if 'disparity_mean' in results:
            print(f"\nDisparity Statistics:")
            print(f"  Mean: {results['disparity_mean']:.2f} pixels")
            print(f"  Std:  {results['disparity_std']:.2f} pixels")
            print(f"  Min:  {results['disparity_min']:.2f} pixels")
            print(f"  Max:  {results['disparity_max']:.2f} pixels")
        
        print(f"{'='*60}\n")
        
        # Save results
        stats_file = os.path.join(output_dir, f'{detector_type}_matching_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Statistics saved to: {stats_file}")
        
        # Save matches for depth recovery
        matches_file = os.path.join(output_dir, f'{detector_type}_matches.pkl')
        matcher.save_matches(kp_left, kp_right, matches, matches_file)
        
        # Visualize
        if fm_cfg.get('visualize', True):
            vis_path = os.path.join(output_dir, f'{detector_type}_matches.png')
            max_display = fm_cfg.get('max_display_matches', 100)
            matcher.visualize_matches(img_left, img_right, kp_left, kp_right,
                                     matches, vis_path, max_matches=max_display)
    
    print("\n✓ Feature matching completed successfully!")
    print(f"Results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()