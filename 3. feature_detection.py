#!/usr/bin/env python3
"""
Feature detection using multiple algorithms: SIFT, ORB, FAST, and Harris.

This script detects features in stereo image pairs using various algorithms
and compares their performance.

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
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_config, ensure_dir, get_image_pair_path, print_config_summary


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
        self.keypoints = None
        self.descriptors = None
        self.computation_time = 0
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the appropriate detector."""
        if self.detector_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif self.detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif self.detector_type == 'fast':
            self.detector = cv2.FastFeatureDetector_create(threshold=10)
        elif self.detector_type == 'harris':
            # Harris is handled separately as it doesn't follow the same interface
            self.detector = None
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
            # FAST only detects keypoints, no descriptors
            keypoints = self.detector.detect(image, None)
            descriptors = None
        else:
            # SIFT and ORB provide both keypoints and descriptors
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        self.computation_time = (time.time() - start_time) * 1000  # Convert to ms
        self.keypoints = keypoints
        self.descriptors = descriptors
        
        return keypoints, descriptors, self.computation_time
    
    def _detect_harris(self, image):
        """
        Detect Harris corners.
        
        Args:
            image (np.array): Input image (grayscale)
            
        Returns:
            tuple: (keypoints, None) - Harris doesn't provide descriptors
        """
        # Harris corner detection parameters
        blockSize = 2
        ksize = 3
        k = 0.04
        
        # Detect corners
        dst = cv2.cornerHarris(image, blockSize, ksize, k)
        
        # Dilate to mark corners
        dst = cv2.dilate(dst, None)
        
        # Threshold for corner detection
        threshold = 0.01 * dst.max()
        
        # Find corner locations
        corner_locations = np.where(dst > threshold)
        
        # Convert to keypoints
        keypoints = []
        for y, x in zip(corner_locations[0], corner_locations[1]):
            kp = cv2.KeyPoint(float(x), float(y), 1)
            keypoints.append(kp)
        
        return keypoints, None
    
    def visualize(self, image, output_path=None, title=None):
        """
        Visualize detected features.
        
        Args:
            image (np.array): Input image
            output_path (str): Path to save visualization
            title (str): Title for the plot
        """
        if self.keypoints is None:
            print("No keypoints to visualize. Run detect() first.")
            return
        
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(
            image, 
            self.keypoints, 
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        
        if title is None:
            title = f'{self.detector_type.upper()} Features: {len(self.keypoints)} keypoints'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add statistics text
        stats_text = f'Features: {len(self.keypoints)}\n'
        stats_text += f'Time: {self.computation_time:.2f} ms'
        plt.text(10, 30, stats_text, fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def compare_detectors(image, detectors, output_dir=None):
    """
    Compare multiple feature detectors on the same image.
    
    Args:
        image (np.array): Input image
        detectors (list): List of detector type strings
        output_dir (str): Directory to save comparison results
        
    Returns:
        dict: Statistics for each detector
    """
    results = {}
    
    print("\n" + "="*60)
    print("Feature Detection Comparison")
    print("="*60 + "\n")
    
    for detector_type in detectors:
        print(f"Running {detector_type.upper()} detector...")
        
        detector = FeatureDetector(detector_type)
        keypoints, descriptors, comp_time = detector.detect(image)
        
        results[detector_type] = {
            'num_features': len(keypoints),
            'computation_time_ms': comp_time,
            'has_descriptors': descriptors is not None,
            'descriptor_size': descriptors.shape[1] if descriptors is not None else 0
        }
        
        print(f"  Features detected: {len(keypoints)}")
        print(f"  Computation time: {comp_time:.2f} ms")
        if descriptors is not None:
            print(f"  Descriptor size: {descriptors.shape[1]}")
        print()
        
        # Save visualization if output directory provided
        if output_dir:
            vis_path = os.path.join(output_dir, f'{detector_type}_features.png')
            detector.visualize(image, vis_path)
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Detector':<12} {'Features':<12} {'Time (ms)':<12} {'Descriptors':<12}")
    print("-"*60)
    
    for detector_type, stats in results.items():
        print(f"{detector_type.upper():<12} {stats['num_features']:<12} "
              f"{stats['computation_time_ms']:<12.2f} "
              f"{'Yes' if stats['has_descriptors'] else 'No':<12}")
    
    print("="*60 + "\n")
    
    return results


def create_comparison_plot(results, output_path=None):
    """
    Create comparison plots for detector performance.
    
    Args:
        results (dict): Results dictionary from compare_detectors
        output_path (str): Path to save the plot
    """
    detectors = list(results.keys())
    num_features = [results[d]['num_features'] for d in detectors]
    comp_times = [results[d]['computation_time_ms'] for d in detectors]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of features
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax1.bar(detectors, num_features, color=colors[:len(detectors)])
    ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Detector Type', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Count Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (d, v) in enumerate(zip(detectors, num_features)):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Computation time
    ax2.bar(detectors, comp_times, color=colors[:len(detectors)])
    ax2.set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Detector Type', fontsize=12, fontweight='bold')
    ax2.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (d, v) in enumerate(zip(detectors, comp_times)):
        ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_statistics(results, output_file):
    """
    Save detection statistics to JSON file.
    
    Args:
        results (dict): Results dictionary
        output_file (str): Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Feature detection on stereo images'
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
    fd_cfg = config['feature_detection']
    images_dir = config['paths']['images_output']
    output_dir = config['paths']['features_output']
    
    # Check if task is enabled
    if not fd_cfg.get('enabled', True):
        print("Feature detection is disabled in config.yaml")
        return
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Get image paths
    img_index = fd_cfg['test_image_index']
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
    
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if left_image is None or right_image is None:
        print("Error: Failed to load images")
        sys.exit(1)
    
    print(f"  Image size: {left_image.shape[1]} x {left_image.shape[0]}\n")
    
    # Process left image
    print("Processing LEFT image:")
    left_output_dir = os.path.join(output_dir, 'cam0')
    ensure_dir(left_output_dir)
    
    visualize = fd_cfg.get('visualize', True)
    left_results = compare_detectors(left_image, fd_cfg['detectors'], 
                                    left_output_dir if visualize else None)
    
    # Process right image
    print("\nProcessing RIGHT image:")
    right_output_dir = os.path.join(output_dir, 'cam1')
    ensure_dir(right_output_dir)
    
    right_results = compare_detectors(right_image, fd_cfg['detectors'],
                                     right_output_dir if visualize else None)
    
    # Create comparison plots
    if visualize:
        print("\nGenerating comparison plots...")
        create_comparison_plot(
            left_results,
            os.path.join(output_dir, 'cam0_comparison.png')
        )
        create_comparison_plot(
            right_results,
            os.path.join(output_dir, 'cam1_comparison.png')
        )
    
    # Save statistics
    if fd_cfg.get('save_statistics', True):
        all_results = {
            'image_index': img_index,
            'cam0': left_results,
            'cam1': right_results
        }
        save_statistics(all_results, os.path.join(output_dir, 'detection_statistics.json'))
    
    # Save text summary
    summary_file = os.path.join(output_dir, 'comparison_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Feature Detection Comparison Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Image Index: {img_index}\n")
        f.write(f"Left Image:  {left_image_path}\n")
        f.write(f"Right Image: {right_image_path}\n")
        f.write(f"Image Size:  {left_image.shape[1]} x {left_image.shape[0]}\n\n")
        
        f.write("LEFT CAMERA (cam0):\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Detector':<12} {'Features':<12} {'Time (ms)':<12} {'Descriptors':<12}\n")
        f.write("-"*60 + "\n")
        for detector_type, stats in left_results.items():
            f.write(f"{detector_type.upper():<12} {stats['num_features']:<12} "
                   f"{stats['computation_time_ms']:<12.2f} "
                   f"{'Yes' if stats['has_descriptors'] else 'No':<12}\n")
        
        f.write("\nRIGHT CAMERA (cam1):\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Detector':<12} {'Features':<12} {'Time (ms)':<12} {'Descriptors':<12}\n")
        f.write("-"*60 + "\n")
        for detector_type, stats in right_results.items():
            f.write(f"{detector_type.upper():<12} {stats['num_features']:<12} "
                   f"{stats['computation_time_ms']:<12.2f} "
                   f"{'Yes' if stats['has_descriptors'] else 'No':<12}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n✓ Feature detection completed successfully!")


if __name__ == '__main__':
    main()