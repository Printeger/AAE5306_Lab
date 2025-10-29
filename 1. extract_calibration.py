#!/usr/bin/env python3
"""
Extract camera calibration parameters from EuRoC MAV dataset YAML file.

This script reads the calibration YAML file from the EuRoC dataset and
extracts the camera intrinsic parameters and baseline for both cameras.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import yaml
import numpy as np
import argparse
import os
import sys
from utils import load_config, ensure_dir, print_config_summary


def load_euroc_calibration(yaml_file):
    """
    Load calibration data from EuRoC YAML file.
    
    Args:
        yaml_file (str): Path to the calibration YAML file
        
    Returns:
        dict: Dictionary containing calibration parameters
    """
    try:
        with open(yaml_file, 'r') as f:
            calib = yaml.safe_load(f)
        return calib
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {yaml_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def extract_camera_parameters(calib, camera_name='cam0'):
    """
    Extract intrinsic parameters for a specific camera.
    
    Args:
        calib (dict): Full calibration dictionary
        camera_name (str): Name of camera ('cam0' or 'cam1')
        
    Returns:
        dict: Dictionary with fx, fy, cx, cy, distortion coefficients
    """
    try:
        cam_data = calib[camera_name]
        
        # Intrinsics: [fx, fy, cx, cy]
        intrinsics = cam_data['intrinsics']
        
        # Distortion coefficients: [k1, k2, p1, p2]
        distortion = cam_data['distortion_coefficients']
        
        # Resolution
        resolution = cam_data['resolution']
        
        params = {
            'fx': intrinsics[0],
            'fy': intrinsics[1],
            'cx': intrinsics[2],
            'cy': intrinsics[3],
            'k1': distortion[0],
            'k2': distortion[1],
            'p1': distortion[2],
            'p2': distortion[3],
            'width': resolution[0],
            'height': resolution[1],
            'camera_model': cam_data.get('camera_model', 'pinhole'),
            'distortion_model': cam_data.get('distortion_model', 'radtan')
        }
        
        return params
        
    except KeyError as e:
        print(f"Error: Missing key in calibration file: {e}")
        sys.exit(1)


def compute_baseline(calib):
    """
    Compute baseline (distance between cameras) from transformation matrix.
    
    Args:
        calib (dict): Full calibration dictionary
        
    Returns:
        float: Baseline distance in meters
    """
    try:
        # T_cn_cnm1 is transformation from cam0 to cam1
        T_cam1_cam0 = np.array(calib['cam1']['T_cn_cnm1'])
        
        # Extract translation vector (last column, first 3 rows)
        translation = T_cam1_cam0[:3, 3]
        
        # Baseline is typically the x-component for horizontal stereo
        # But we compute the norm for generality
        baseline = np.linalg.norm(translation)
        
        # For horizontal stereo, baseline should be approximately the x-component
        baseline_x = abs(translation[0])
        
        return baseline, baseline_x, translation
        
    except KeyError as e:
        print(f"Error: Missing transformation matrix: {e}")
        sys.exit(1)


def print_calibration_summary(cam0_params, cam1_params, baseline, baseline_x, translation):
    """
    Print formatted calibration summary.
    
    Args:
        cam0_params (dict): Camera 0 parameters
        cam1_params (dict): Camera 1 parameters
        baseline (float): Baseline distance (norm)
        baseline_x (float): Baseline x-component
        translation (np.array): Translation vector
    """
    print("\n" + "="*60)
    print("EuRoC MAV Dataset - Camera Calibration Parameters")
    print("="*60)
    
    print("\n--- Camera 0 (Left) Intrinsics ---")
    print(f"Focal Length (fx):    {cam0_params['fx']:.4f} pixels")
    print(f"Focal Length (fy):    {cam0_params['fy']:.4f} pixels")
    print(f"Principal Point (cx): {cam0_params['cx']:.4f} pixels")
    print(f"Principal Point (cy): {cam0_params['cy']:.4f} pixels")
    print(f"Resolution:           {cam0_params['width']} x {cam0_params['height']}")
    print(f"Camera Model:         {cam0_params['camera_model']}")
    print(f"Distortion Model:     {cam0_params['distortion_model']}")
    print(f"\nDistortion Coefficients:")
    print(f"  k1: {cam0_params['k1']:.6f}")
    print(f"  k2: {cam0_params['k2']:.6f}")
    print(f"  p1: {cam0_params['p1']:.6f}")
    print(f"  p2: {cam0_params['p2']:.6f}")
    
    print("\n--- Camera 1 (Right) Intrinsics ---")
    print(f"Focal Length (fx):    {cam1_params['fx']:.4f} pixels")
    print(f"Focal Length (fy):    {cam1_params['fy']:.4f} pixels")
    print(f"Principal Point (cx): {cam1_params['cx']:.4f} pixels")
    print(f"Principal Point (cy): {cam1_params['cy']:.4f} pixels")
    print(f"Resolution:           {cam1_params['width']} x {cam1_params['height']}")
    
    print("\n--- Stereo Configuration ---")
    print(f"Baseline (norm):      {baseline:.6f} meters ({baseline*1000:.2f} mm)")
    print(f"Baseline (x-axis):    {baseline_x:.6f} meters ({baseline_x*1000:.2f} mm)")
    print(f"Translation vector:   [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}] meters")
    
    print("\n--- Depth Range Estimation ---")
    # Assuming minimum disparity of 1 pixel
    max_depth = (cam0_params['fx'] * baseline) / 1.0
    # Assuming maximum disparity of 100 pixels
    min_depth = (cam0_params['fx'] * baseline) / 100.0
    print(f"Approximate depth range (1-100 px disparity):")
    print(f"  Minimum depth: {min_depth:.2f} meters")
    print(f"  Maximum depth: {max_depth:.2f} meters")
    
    print("\n" + "="*60 + "\n")


def save_calibration_for_opencv(cam0_params, cam1_params, baseline, output_file):
    """
    Save calibration in OpenCV-compatible format.
    
    Args:
        cam0_params (dict): Camera 0 parameters
        cam1_params (dict): Camera 1 parameters
        baseline (float): Baseline distance
        output_file (str): Path to output YAML file
    """
    # Create camera matrix format for OpenCV
    calibration_data = {
        'cam0': {
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [
                    cam0_params['fx'], 0.0, cam0_params['cx'],
                    0.0, cam0_params['fy'], cam0_params['cy'],
                    0.0, 0.0, 1.0
                ]
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 4,
                'data': [
                    cam0_params['k1'],
                    cam0_params['k2'],
                    cam0_params['p1'],
                    cam0_params['p2']
                ]
            },
            'resolution': [cam0_params['width'], cam0_params['height']]
        },
        'cam1': {
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [
                    cam1_params['fx'], 0.0, cam1_params['cx'],
                    0.0, cam1_params['fy'], cam1_params['cy'],
                    0.0, 0.0, 1.0
                ]
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 4,
                'data': [
                    cam1_params['k1'],
                    cam1_params['k2'],
                    cam1_params['p1'],
                    cam1_params['p2']
                ]
            },
            'resolution': [cam1_params['width'], cam1_params['height']]
        },
        'stereo': {
            'baseline': float(baseline),
            'baseline_mm': float(baseline * 1000)
        }
    }
    
    # Save to YAML
    try:
        with open(output_file, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        print(f"Calibration saved to: {output_file}")
    except Exception as e:
        print(f"Error saving calibration file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract camera calibration from EuRoC MAV dataset'
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
    
    # Get paths from config
    input_yaml = config['paths']['euroc_calibration']
    output_yaml = config['paths']['calibration_output']
    
    # Check if task is enabled
    if not config.get('calibration', {}).get('enabled', True):
        print("Calibration extraction is disabled in config.yaml")
        return
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Check if input file exists
    if not os.path.exists(input_yaml):
        print(f"Error: Input file not found: {input_yaml}")
        print("Please download the EuRoC calibration file first.")
        print("Download from: http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/")
        sys.exit(1)
    
    # Load calibration
    print(f"Loading calibration from: {input_yaml}")
    calib = load_euroc_calibration(input_yaml)
    
    # Extract parameters for both cameras
    cam0_params = extract_camera_parameters(calib, 'cam0')
    cam1_params = extract_camera_parameters(calib, 'cam1')
    
    # Compute baseline
    baseline, baseline_x, translation = compute_baseline(calib)
    
    # Print summary
    print_calibration_summary(cam0_params, cam1_params, baseline, baseline_x, translation)
    
    # Save for OpenCV
    ensure_dir(os.path.dirname(output_yaml))
    save_calibration_for_opencv(cam0_params, cam1_params, baseline, output_yaml)
    
    print("\n✓ Calibration extraction completed successfully!")
    print(f"Output saved to: {output_yaml}\n")


if __name__ == '__main__':
    main()