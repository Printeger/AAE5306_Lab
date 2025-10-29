#!/usr/bin/env python3
"""
Utility functions for AAE5306 Lab.

This module provides common utilities including configuration loading,
path management, and helper functions.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import yaml
import os
import sys
from pathlib import Path


def load_config(config_file='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        print("Please ensure config.yaml exists in the project root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    os.makedirs(directory, exist_ok=True)


def get_image_pair_path(images_dir, index, camera='cam0'):
    """
    Get path to a specific image pair.
    
    Args:
        images_dir (str): Base directory for extracted images
        index (int): Image index (0-based)
        camera (str): Camera name ('cam0' or 'cam1')
        
    Returns:
        str: Path to the image file
    """
    filename = f"{index:06d}.png"
    return os.path.join(images_dir, camera, filename)


def validate_paths(config):
    """
    Validate that required input paths exist.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: List of validation errors (empty if all valid)
    """
    errors = []
    
    # Check calibration file (only if calibration is enabled)
    if config.get('calibration', {}).get('enabled', True):
        calib_file = config['paths']['euroc_calibration']
        if not os.path.exists(calib_file):
            errors.append(f"EuRoC calibration file not found: {calib_file}")
    
    # Check rosbag file (only if extraction is enabled)
    if config.get('image_extraction', {}).get('enabled', True):
        bag_file = config['paths']['rosbag_file']
        if not os.path.exists(bag_file):
            errors.append(f"ROS bag file not found: {bag_file}")
    
    return errors


def print_config_summary(config):
    """
    Print a summary of the current configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "="*70)
    print("Configuration Summary")
    print("="*70)
    
    # Paths
    print("\nPaths:")
    print(f"  EuRoC Calibration: {config['paths']['euroc_calibration']}")
    print(f"  ROS Bag File:      {config['paths']['rosbag_file']}")
    print(f"  Output Base:       {config['paths']['output_base']}")
    
    # Enabled tasks
    print("\nEnabled Tasks:")
    print(f"  1. Calibration Extraction: {config.get('calibration', {}).get('enabled', True)}")
    print(f"  2. Image Extraction:       {config.get('image_extraction', {}).get('enabled', True)}")
    print(f"  3. Feature Detection:      {config.get('feature_detection', {}).get('enabled', True)}")
    print(f"  4. Feature Matching:       {config.get('feature_matching', {}).get('enabled', True)}")
    print(f"  5. Depth Recovery:         {config.get('depth_recovery', {}).get('enabled', True)}")
    
    # Image extraction settings
    if config.get('image_extraction', {}).get('enabled', True):
        ie_cfg = config['image_extraction']
        print("\nImage Extraction Settings:")
        print(f"  Skip Frames:  {ie_cfg['skip_frames']}")
        print(f"  Max Pairs:    {ie_cfg['max_pairs']}")
    
    # Feature detection settings
    if config.get('feature_detection', {}).get('enabled', True):
        fd_cfg = config['feature_detection']
        print("\nFeature Detection Settings:")
        print(f"  Detectors:    {', '.join(fd_cfg['detectors'])}")
        print(f"  Test Image:   {fd_cfg['test_image_index']}")
    
    # Feature matching settings
    if config.get('feature_matching', {}).get('enabled', True):
        fm_cfg = config['feature_matching']
        print("\nFeature Matching Settings:")
        print(f"  Detectors:    {', '.join(fm_cfg['detectors'])}")
        print(f"  SIFT Ratio:   {fm_cfg['sift']['ratio_threshold']}")
        print(f"  ORB Ratio:    {fm_cfg['orb']['ratio_threshold']}")
    
    # Depth recovery settings
    if config.get('depth_recovery', {}).get('enabled', True):
        dr_cfg = config['depth_recovery']
        print("\nDepth Recovery Settings:")
        print(f"  Detector:     {dr_cfg['detector']}")
        print(f"  Depth Range:  {dr_cfg['min_depth']} - {dr_cfg['max_depth']} m")
    
    print("="*70 + "\n")


def get_config_value(config, *keys, default=None):
    """
    Safely get nested configuration value.
    
    Args:
        config (dict): Configuration dictionary
        *keys: Nested keys to access
        default: Default value if key doesn't exist
        
    Returns:
        Value at the specified key path or default
        
    Example:
        get_config_value(config, 'feature_matching', 'sift', 'ratio_threshold')
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
