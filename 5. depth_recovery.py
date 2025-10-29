#!/usr/bin/env python3
"""
Recover 3D depth from stereo disparity using matched features.

This script computes depth from feature matches and generates 3D point clouds.

Author: AAE5306 Teaching Team
Date: October 29, 2025
"""

import numpy as np
import cv2
import argparse
import os
import sys
import yaml
import pickle
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import load_config, ensure_dir, get_image_pair_path, print_config_summary


class DepthRecovery:
    """Recover depth from stereo disparity."""
    
    def __init__(self, calibration_file):
        """
        Initialize depth recovery with camera calibration.
        
        Args:
            calibration_file (str): Path to calibration YAML file
        """
        self.calibration = self._load_calibration(calibration_file)
        self.points_3d = None
        self.colors = None
        self.disparities = None
    
    def _load_calibration(self, calibration_file):
        """Load camera calibration parameters."""
        try:
            with open(calibration_file, 'r') as f:
                calib = yaml.safe_load(f)
            
            # Extract parameters
            cam0_matrix = np.array(calib['cam0']['camera_matrix']['data']).reshape(3, 3)
            
            calibration = {
                'fx': cam0_matrix[0, 0],
                'fy': cam0_matrix[1, 1],
                'cx': cam0_matrix[0, 2],
                'cy': cam0_matrix[1, 2],
                'baseline': calib['stereo']['baseline'],
                'width': calib['cam0']['resolution'][0],
                'height': calib['cam0']['resolution'][1]
            }
            
            print("\nCamera Calibration Loaded:")
            print(f"  Focal length (fx): {calibration['fx']:.2f} pixels")
            print(f"  Focal length (fy): {calibration['fy']:.2f} pixels")
            print(f"  Principal point: ({calibration['cx']:.2f}, {calibration['cy']:.2f})")
            print(f"  Baseline: {calibration['baseline']:.6f} m ({calibration['baseline']*1000:.2f} mm)")
            print(f"  Resolution: {calibration['width']} x {calibration['height']}\n")
            
            return calibration
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            sys.exit(1)
    
    def compute_depth(self, pts_left, pts_right, min_depth=0.5, max_depth=15.0):
        """
        Compute depth from matched points.
        
        Args:
            pts_left (np.array): Matched points in left image (Nx2)
            pts_right (np.array): Matched points in right image (Nx2)
            min_depth (float): Minimum valid depth (meters)
            max_depth (float): Maximum valid depth (meters)
            
        Returns:
            dict: Dictionary containing 3D points and statistics
        """
        print(f"Computing depth from {len(pts_left)} matched points...")
        
        # Compute disparity
        disparities = pts_left[:, 0] - pts_right[:, 0]
        
        # Filter invalid disparities (must be positive for rectified stereo)
        valid_mask = disparities > 0
        
        pts_left_valid = pts_left[valid_mask]
        pts_right_valid = pts_right[valid_mask]
        disparities_valid = disparities[valid_mask]
        
        print(f"  Valid disparities: {len(disparities_valid)} / {len(disparities)}")
        
        # Compute depth: Z = (f * baseline) / disparity
        fx = self.calibration['fx']
        baseline = self.calibration['baseline']
        
        depths = (fx * baseline) / disparities_valid
        
        # Filter by depth range
        depth_mask = (depths >= min_depth) & (depths <= max_depth)
        
        pts_left_final = pts_left_valid[depth_mask]
        disparities_final = disparities_valid[depth_mask]
        depths_final = depths[depth_mask]
        
        print(f"  Valid depth range [{min_depth}, {max_depth}] m: {len(depths_final)} points")
        
        # Compute 3D coordinates
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        
        cx = self.calibration['cx']
        cy = self.calibration['cy']
        fy = self.calibration['fy']
        
        X = (pts_left_final[:, 0] - cx) * depths_final / fx
        Y = (pts_left_final[:, 1] - cy) * depths_final / fy
        Z = depths_final
        
        # Stack into Nx3 array
        points_3d = np.column_stack([X, Y, Z])
        
        self.points_3d = points_3d
        self.disparities = disparities_final
        
        # Compute statistics
        stats = {
            'num_total_matches': len(pts_left),
            'num_valid_disparity': len(disparities_valid),
            'num_valid_depth': len(depths_final),
            'min_depth': float(np.min(depths_final)),
            'max_depth': float(np.max(depths_final)),
            'mean_depth': float(np.mean(depths_final)),
            'median_depth': float(np.median(depths_final)),
            'std_depth': float(np.std(depths_final)),
            'min_disparity': float(np.min(disparities_final)),
            'max_disparity': float(np.max(disparities_final)),
            'mean_disparity': float(np.mean(disparities_final))
        }
        
        print(f"\nDepth Statistics:")
        print(f"  Number of 3D points: {stats['num_valid_depth']}")
        print(f"  Depth range: [{stats['min_depth']:.2f}, {stats['max_depth']:.2f}] m")
        print(f"  Mean depth: {stats['mean_depth']:.2f} m")
        print(f"  Median depth: {stats['median_depth']:.2f} m")
        print(f"  Std depth: {stats['std_depth']:.2f} m")
        print(f"  Disparity range: [{stats['min_disparity']:.2f}, {stats['max_disparity']:.2f}] px\n")
        
        return stats
    
    def create_colored_pointcloud(self, image):
        """
        Create colored point cloud from image.
        
        Args:
            image (np.array): Left image for coloring
        """
        if self.points_3d is None:
            print("Error: No 3D points computed. Run compute_depth() first.")
            return
        
        # For grayscale images, use depth-based coloring
        if len(image.shape) == 2:
            # Normalize depths to [0, 1]
            depths_normalized = (self.points_3d[:, 2] - self.points_3d[:, 2].min()) / \
                              (self.points_3d[:, 2].max() - self.points_3d[:, 2].min())
            
            # Use colormap (jet)
            import matplotlib.cm as cm
            colormap = cm.get_cmap('jet')
            colors = colormap(depths_normalized)[:, :3]  # RGB only
            
        else:
            # Use actual image colors (not implemented for this lab)
            colors = np.ones((len(self.points_3d), 3)) * 0.5
        
        self.colors = (colors * 255).astype(np.uint8)
    
    def save_pointcloud(self, output_file):
        """
        Save point cloud to PLY file.
        
        Args:
            output_file (str): Output PLY file path
        """
        if self.points_3d is None:
            print("Error: No 3D points to save.")
            return
        
        points = self.points_3d
        colors = self.colors if self.colors is not None else np.ones((len(points), 3)) * 128
        
        # Write PLY file
        with open(output_file, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Data
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
                f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
        
        print(f"Point cloud saved to: {output_file}")
    
    def visualize_depth_map(self, image_shape, pts_left, output_path=None):
        """
        Visualize depth as a colored map.
        
        Args:
            image_shape (tuple): Shape of the original image (height, width)
            pts_left (np.array): Left image points
            output_path (str): Path to save visualization
        """
        if self.points_3d is None:
            print("Error: No depth data to visualize.")
            return
        
        # Create depth map image
        depth_map = np.zeros(image_shape, dtype=np.float32)
        
        # Fill in depth values at feature locations
        for i, (pt, depth) in enumerate(zip(pts_left, self.points_3d[:, 2])):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                depth_map[y, x] = depth
        
        # Create colored visualization
        plt.figure(figsize=(12, 8))
        
        # Use jet colormap
        plt.imshow(depth_map, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Depth (meters)', shrink=0.8)
        plt.title('Depth Map from Stereo Matching', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Depth map saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_pointcloud_2d(self, output_path=None):
        """
        Create 2D projection visualization of point cloud.
        
        Args:
            output_path (str): Path to save visualization
        """
        if self.points_3d is None:
            print("Error: No point cloud to visualize.")
            return
        
        fig = plt.figure(figsize=(15, 5))
        
        # Top view (X-Z plane)
        ax1 = fig.add_subplot(131)
        scatter1 = ax1.scatter(self.points_3d[:, 0], self.points_3d[:, 2],
                              c=self.points_3d[:, 2], cmap='jet', s=1)
        ax1.set_xlabel('X (m)', fontweight='bold')
        ax1.set_ylabel('Z (m)', fontweight='bold')
        ax1.set_title('Top View (X-Z)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Side view (Y-Z plane)
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(self.points_3d[:, 1], self.points_3d[:, 2],
                              c=self.points_3d[:, 2], cmap='jet', s=1)
        ax2.set_xlabel('Y (m)', fontweight='bold')
        ax2.set_ylabel('Z (m)', fontweight='bold')
        ax2.set_title('Side View (Y-Z)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Front view (X-Y plane)
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(self.points_3d[:, 0], self.points_3d[:, 1],
                              c=self.points_3d[:, 2], cmap='jet', s=1)
        ax3.set_xlabel('X (m)', fontweight='bold')
        ax3.set_ylabel('Y (m)', fontweight='bold')
        ax3.set_title('Front View (X-Y)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Add colorbar
        fig.colorbar(scatter3, ax=[ax1, ax2, ax3], label='Depth (m)', shrink=0.6)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Point cloud 2D projection saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Recover depth from stereo matches'
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
    dr_cfg = config['depth_recovery']
    images_dir = config['paths']['images_output']
    matches_dir = config['paths']['matches_output']
    calibration_file = config['paths']['calibration_output']
    output_dir = config['paths']['depth_output']
    
    # Check if task is enabled
    if not dr_cfg.get('enabled', True):
        print("Depth recovery is disabled in config.yaml")
        return
    
    # Print configuration summary
    if config.get('advanced', {}).get('verbose', True):
        print_config_summary(config)
    
    # Get detector and parameters
    detector_type = dr_cfg['detector']
    min_depth = dr_cfg['min_depth']
    max_depth = dr_cfg['max_depth']
    
    # Get image index from feature matching config
    img_index = config['feature_matching']['test_image_index']
    
    # Construct file paths
    matches_file = os.path.join(matches_dir, f'{detector_type}_matches.pkl')
    left_image_path = get_image_pair_path(images_dir, img_index, 'cam0')
    
    # Check files exist
    if not os.path.exists(matches_file):
        print(f"Error: Matches file not found: {matches_file}")
        print(f"Please run feature matching first (Task 4)")
        sys.exit(1)
    
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file not found: {calibration_file}")
        print(f"Please run calibration extraction first (Task 1)")
        sys.exit(1)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Load matches
    print(f"Loading matches from: {matches_file}")
    with open(matches_file, 'rb') as f:
        match_data = pickle.load(f)
    
    pts_left = match_data['pts_left']
    pts_right = match_data['pts_right']
    
    print(f"  Detector type: {detector_type}")
    print(f"  Number of matches: {len(pts_left)}\n")
    
    # Check minimum matches requirement
    min_matches = config.get('advanced', {}).get('min_matches_required', 50)
    if len(pts_left) < min_matches:
        print(f"Warning: Only {len(pts_left)} matches found, minimum recommended is {min_matches}")
    
    # Load left image
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    if left_image is None:
        print(f"Error: Could not load image: {left_image_path}")
        sys.exit(1)
    
    # Initialize depth recovery
    depth_recovery = DepthRecovery(calibration_file)
    
    # Compute depth
    stats = depth_recovery.compute_depth(
        pts_left,
        pts_right,
        min_depth=min_depth,
        max_depth=max_depth
    )
    
    # Create colored point cloud
    if dr_cfg.get('use_depth_coloring', True):
        depth_recovery.create_colored_pointcloud(left_image)
    
    # Save point cloud
    pc_format = config.get('advanced', {}).get('pointcloud_format', 'ply')
    ply_file = os.path.join(output_dir, f'point_cloud_{detector_type}.{pc_format}')
    depth_recovery.save_pointcloud(ply_file)
    
    # Save statistics
    stats['detector_type'] = detector_type
    stats['image_index'] = img_index
    stats['min_depth_threshold'] = min_depth
    stats['max_depth_threshold'] = max_depth
    
    stats_file = os.path.join(output_dir, f'depth_statistics_{detector_type}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved to: {stats_file}")
    
    # Visualizations
    if dr_cfg.get('visualize', True):
        print("\nGenerating visualizations...")
        
        # Depth map
        if dr_cfg.get('generate_depth_map', True):
            depth_map_file = os.path.join(output_dir, f'depth_map_{detector_type}.png')
            
            # Get valid points that contributed to depth
            disparities = pts_left[:, 0] - pts_right[:, 0]
            valid_mask = disparities > 0
            pts_left_valid = pts_left[valid_mask]
            
            # Apply depth range filter
            fx = depth_recovery.calibration['fx']
            baseline = depth_recovery.calibration['baseline']
            depths = (fx * baseline) / disparities[valid_mask]
            depth_mask = (depths >= min_depth) & (depths <= max_depth)
            pts_left_final = pts_left_valid[depth_mask]
            
            depth_recovery.visualize_depth_map(
                left_image.shape,
                pts_left_final,
                depth_map_file
            )
        
        # Point cloud 2D projections
        if dr_cfg.get('generate_2d_projections', True):
            pc_2d_file = os.path.join(output_dir, f'pointcloud_2d_{detector_type}.png')
            depth_recovery.visualize_pointcloud_2d(pc_2d_file)
    
    print("\n✓ Depth recovery completed successfully!")
    print(f"  3D points: {stats['num_valid_depth']}")
    print(f"  Mean depth: {stats['mean_depth']:.2f} m")
    print(f"  Output directory: {output_dir}\n")


if __name__ == '__main__':
    main()