#!/usr/bin/env python3
"""
Main runner script for AAE5306 Lab pipeline.

This script runs all tasks in sequence according to the configuration file.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import argparse
import subprocess
import sys
import os
from utils import load_config, print_config_summary, validate_paths


def run_task(task_name, script_path, config_file='config.yaml'):
    """
    Run a single task script.
    
    Args:
        task_name (str): Name of the task for display
        script_path (str): Path to the Python script
        config_file (str): Path to configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print(f"Running: {task_name}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path, '--config', config_file],
            check=True,
            capture_output=False
        )
        print(f"✓ {task_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {task_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {task_name} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run AAE5306 Lab complete pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Tasks to run (default: 1 2 3 4 5). Example: --tasks 3 4 5'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip path validation checks'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("="*70)
    print("AAE5306 Lab - Stereo Vision Processing Pipeline")
    print("="*70)
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Validate paths (unless skipped)
    if not args.skip_validation:
        print("\nValidating paths...")
        errors = validate_paths(config)
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"  ✗ {error}")
            print("\nPlease fix the errors or use --skip-validation to continue anyway.")
            sys.exit(1)
        print("✓ All required paths validated\n")
    
    # Define tasks
    tasks = {
        1: {
            'name': 'Task 1: Extract Camera Calibration',
            'script': '1. extract_calibration.py',
            'enabled_key': ['calibration', 'enabled']
        },
        2: {
            'name': 'Task 2: Extract Stereo Images',
            'script': '2. extract_images.py',
            'enabled_key': ['image_extraction', 'enabled']
        },
        3: {
            'name': 'Task 3: Feature Detection',
            'script': '3. feature_detection.py',
            'enabled_key': ['feature_detection', 'enabled']
        },
        4: {
            'name': 'Task 4: Feature Matching',
            'script': '4. feature_matching.py',
            'enabled_key': ['feature_matching', 'enabled']
        },
        5: {
            'name': 'Task 5: Depth Recovery',
            'script': '5. depth_recovery.py',
            'enabled_key': ['depth_recovery', 'enabled']
        }
    }
    
    # Check which tasks are enabled
    tasks_to_run = []
    for task_num in sorted(args.tasks):
        if task_num not in tasks:
            print(f"Warning: Task {task_num} does not exist, skipping...")
            continue
        
        task = tasks[task_num]
        
        # Check if task is enabled in config
        enabled = config
        for key in task['enabled_key']:
            enabled = enabled.get(key, True)
        
        if enabled:
            tasks_to_run.append((task_num, task))
        else:
            print(f"⊘ {task['name']} is disabled in config.yaml, skipping...")
    
    if not tasks_to_run:
        print("\nNo tasks to run. All selected tasks are disabled in config.yaml")
        return
    
    # Run tasks
    print(f"\nWill run {len(tasks_to_run)} task(s):")
    for task_num, task in tasks_to_run:
        print(f"  {task_num}. {task['name']}")
    print()
    
    results = {}
    for task_num, task in tasks_to_run:
        success = run_task(task['name'], task['script'], args.config)
        results[task_num] = success
        
        if not success:
            print(f"\n{'='*70}")
            print(f"Pipeline stopped at Task {task_num} due to error")
            print(f"{'='*70}")
            break
    
    # Print summary
    print("\n" + "="*70)
    print("Pipeline Execution Summary")
    print("="*70)
    
    for task_num, task in tasks_to_run:
        if task_num in results:
            status = "✓ SUCCESS" if results[task_num] else "✗ FAILED"
            print(f"Task {task_num}: {status}")
        else:
            print(f"Task {task_num}: ⊘ NOT RUN")
    
    # Overall result
    all_success = all(results.values())
    if all_success:
        print("\n🎉 All tasks completed successfully!")
        print("\nResults location:")
        print(f"  - Calibration: {config['paths']['calibration_output']}")
        print(f"  - Images: {config['paths']['images_output']}")
        print(f"  - Features: {config['paths']['features_output']}")
        print(f"  - Matches: {config['paths']['matches_output']}")
        print(f"  - Depth: {config['paths']['depth_output']}")
    else:
        print("\n⚠ Some tasks failed. Please check the error messages above.")
        sys.exit(1)
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
