# AAE5306 Lab: Stereo Vision Processing with EuRoC MAV Dataset

## Overview

This laboratory implements a complete stereo vision processing pipeline using the EuRoC MAV drone dataset, including camera calibration extraction, image extraction, feature detection, feature matching, and depth recovery.

## Project Structure

```
AAE5306_Lab
	│   ├── README.md
	│   ├── run_demo.sh
	│   ├── run_pipeline.py
	│   ├── setup.sh
	│   ├── src
	│   │   └── aae5306_stereo_vision
	│   │       ├── CMakeLists.txt
	│   │       ├── config
	│   │       │   └── stereo_params.yaml
	│   │       ├── depth_recovery_node.py
	│   │       ├── feature_detection_node.py
	│   │       ├── feature_matching_node.py
	│   │       ├── launch
	│   │       │   ├── depth_recovery.launch
	│   │       │   ├── feature_detection.launch
	│   │       │   ├── feature_matching.launch
	│   │       │   ├── stereo_pipeline.launch
	│   │       │   └── stereo_visualization.launch
	│   │       ├── msg
	│   │       │   ├── DepthStats.msg
	│   │       │   ├── FeatureStats.msg
	│   │       │   └── MatchStats.msg
	│   │       ├── package.xml
	│   │       └── rviz
	│   │           └── stereo_vision.rviz
	│   ├── utils.py
	│   └── yaml_parser.py
```

## Requirements
- ROS noetic
- Ubuntu 20.04
- Python 3.7+

### Dependencies
```bash
# Core dependencies
pip install numpy
pip install opencv-contrib-python  # Includes SIFT
pip install matplotlib
pip install pyyaml

# ROS-related (only for Task 2)
pip install rospkg
# Or use system ROS installation: sudo apt-get install ros-noetic-cv-bridge
```

### System Requirements
- ROS Noetic is required for Task 2 (image extraction)
- Windows users can skip Task 2 and use pre-extracted images

## Configuration

<!-- All parameters are configured in the `config.yaml` file. Main configuration items include:

### 1. Paths Configuration
```yaml
paths:
  euroc_calibration: "data/euroc/MH_01_easy.yaml"
  rosbag_file: "data/euroc/rosbags/MH_01_easy.bag"
  output_base: "results"
  # ... other paths
``` -->


## Usage

### Preparation

#### 1. First, create the workspace and clone the repository:
```bash
cd ~/
mkdir -p aae5306_ws/src
cd aae5306_ws/src
git clone https://github.com/Printeger/AAE5306_Lab.git
cd ..  # go to workspace root
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3  # using default python3

source devel/setup.bash
```
#### Download EuRoC Dataset
```bash
# Create data directory
cd aae5306_ws/
mkdir -p data/rosbag

# Download ROS bag file
cd data/rosbags
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
#after download, you should see a .bag file in the folder
ls
#expected output:
# MH_01_easy.bag
```

#### Configure Parameters
Edit the `config.yaml` file to adjust parameters as needed.
```yaml
# Sample modification
processing:
  detector:
    type: orb  # detector type: sift/orb/harris/fast are supported
    publish_rate: 10.0
    visualize: true
  matching:
    ratio_threshold: 0.75
    epipolar_threshold: 2.0
    max_delay: 0.1
    visualize: true
  depth:
    min_depth: 0.5
    max_depth: 15.0
    max_delay: 0.1
    visualize: true
```
### Execution Steps

#### Run the code
```bash
#run with different task
roslaunch aae5306_stereo_vision feature_detection.launch 
# roslaunch aae5306_stereo_vision feature_matching.launch
# roslaunch aae5306_stereo_vision depth_recovery.launch 
# roslaunch aae5306_stereo_vision stereo_visualization.launch 
```


#### Play ROSBAG
```bash 
# Open a new terminal
cd ~/aae5306_ws/data/rosbag
rosbag play MH_01_easy .bag -r 0.5 --clock
```

<!-- ---

#### Task 2: Extract Stereo Image Pairs

**Input:**
- `data/euroc/rosbags/MH_01_easy.bag` - ROS bag file

**Output:**
- `data/extracted_images/cam0/` - Left camera images (000000.png, 000001.png, ...)
- `data/extracted_images/cam1/` - Right camera images
- `data/extracted_images/timestamps.txt` - Timestamp file

**Run:**
```bash
python "2. extract_images.py"
```

**Configuration in config.yaml:**
```yaml
image_extraction:
  skip_frames: 10      # Adjust sampling frequency
  max_pairs: 300       # Adjust extraction quantity
  cam0_topic: "/cam0/image_raw"
  cam1_topic: "/cam1/image_raw"
  info_only: false     # Set to true to only view bag info
```

**Note:** Windows users who cannot install ROS can:
- Use a Linux virtual machine for this task
- Or obtain pre-extracted images from the instructor

---

#### Task 3: Feature Detection

**Input:**
- Extracted image pairs (from Task 2)
- Specify which image pair to use via `test_image_index`

**Output:**
- `results/features/cam0/` - Left image feature visualizations
  - `sift_features.png`
  - `orb_features.png`
  - `fast_features.png`
  - `harris_features.png`
- `results/features/cam1/` - Right image feature visualizations
- `results/features/cam0_comparison.png` - Performance comparison chart
- `results/features/cam1_comparison.png`
- `results/features/detection_statistics.json` - Statistics data
- `results/features/comparison_summary.txt` - Text summary

**Run:**
```bash
python "3. feature_detection.py"
```

**Configuration in config.yaml:**
```yaml
feature_detection:
  detectors:           # Select detectors to use
    - sift
    - orb
    - fast
    - harris
  test_image_index: 50 # Select test image (0-299)
  visualize: true      # Whether to generate visualizations
  save_statistics: true
```

**Output Explanation:**
- **num_features**: Number of detected feature points
- **computation_time_ms**: Computation time (milliseconds)
- **has_descriptors**: Whether descriptors are included (FAST and Harris don't have them)

---

#### Task 4: Feature Matching

**Input:**
- Extracted image pairs (from Task 2)
- Use the same `test_image_index` as Task 3

**Output:**
- `results/matches/sift_matches.pkl` - SIFT matching data (for Task 5)
- `results/matches/orb_matches.pkl` - ORB matching data
- `results/matches/sift_matching_stats.json` - Matching statistics
- `results/matches/orb_matching_stats.json`
- `results/matches/sift_matches.png` - Matching visualization
- `results/matches/orb_matches.png`

**Run:**
```bash
python "4. feature_matching.py"
```

**Configuration in config.yaml:**
```yaml
feature_matching:
  detectors:
    - sift             # Only SIFT and ORB support matching
    - orb
  test_image_index: 50 # Should match Task 3
  
  sift:
    ratio_threshold: 0.75      # Lowe's ratio test (0.7-0.8 recommended)
    epipolar_threshold: 2.0    # Epipolar constraint (pixels)
  
  orb:
    ratio_threshold: 0.75
    epipolar_threshold: 2.0
  
  visualize: true
  max_display_matches: 100     # Maximum matches to display in visualization
```

**Parameter Tuning Suggestions:**
- **ratio_threshold**: Lower values (e.g., 0.7) give more reliable but fewer matches
- **epipolar_threshold**: For rectified stereo images, 2.0 pixels is reasonable
- Re-run after adjusting parameters to observe matching quality changes

**Output Explanation:**
- **num_initial_matches**: Initial match count
- **num_ratio_filtered**: Match count after ratio test
- **num_final_matches**: Final match count after epipolar constraint
- **disparity_mean/std**: Disparity statistics

---

#### Task 5: Depth Recovery

**Input:**
- `results/matches/{detector}_matches.pkl` - Matching data (from Task 4)
- `config/euroc_calibration.yaml` - Calibration data (from Task 1)
- Left image (for point cloud coloring)

**Output:**
- `results/depth/point_cloud_{detector}.ply` - 3D point cloud (viewable with CloudCompare/MeshLab)
- `results/depth/depth_statistics_{detector}.json` - Depth statistics
- `results/depth/depth_map_{detector}.png` - Depth map visualization
- `results/depth/pointcloud_2d_{detector}.png` - Point cloud 2D projections

**Run:**
```bash
python "5. depth_recovery.py"
```

**Configuration in config.yaml:**
```yaml
depth_recovery:
  detector: sift       # Which detector's matching results to use
  min_depth: 0.5      # Filter depths less than this value (meters)
  max_depth: 15.0     # Filter depths greater than this value (meters)
  
  use_depth_coloring: true  # Use depth coloring for point cloud
  
  visualize: true
  generate_depth_map: true
  generate_2d_projections: true
```

**Parameter Tuning Suggestions:**
- Adjust **min_depth** and **max_depth** based on scene
- Indoor scenes: 0.5-10 meters
- Outdoor scenes: 1-50 meters
- Adjusting range can filter noise points

**Output Explanation:**
- **num_valid_depth**: Number of valid 3D points
- **mean_depth**: Average depth (meters)
- **depth_range**: Depth range [min, max]
- **PLY file**: Can be opened with:
  - CloudCompare (recommended)
  - MeshLab
  - Open3D

---

### Run Complete Pipeline

Run all tasks in sequence:

```bash
# Task 1: Extract calibration
python "1. extract_calibration.py"

# Task 2: Extract images (requires ROS)
python "2. extract_images.py"

# Task 3: Feature detection
python "3. feature_detection.py"

# Task 4: Feature matching
python "4. feature_matching.py"

# Task 5: Depth recovery
python "5. depth_recovery.py"
``` -->

<!-- ## Configuration File Details

### Input/Output Summary

| Task | Input | Output | Configurable Parameters |
|------|------|--------|------------------------|
| 1. Calibration Extraction | `MH_01_easy.yaml` | `euroc_calibration.yaml` | None |
| 2. Image Extraction | `MH_01_easy.bag` | `extracted_images/` | `skip_frames`, `max_pairs`, `topics` |
| 3. Feature Detection | Image pairs | Visualizations, statistics | `detectors`, `test_image_index` |
| 4. Feature Matching | Image pairs | Match data, visualizations | `ratio_threshold`, `epipolar_threshold` |
| 5. Depth Recovery | Match data, calibration | Point cloud, depth map | `min_depth`, `max_depth`, `detector` | -->

### Key Parameters

#### test_image_index
- **Range**: 0 to (max_pairs - 1)
- **Default**: 50
- **Description**: Select which image pair to use for feature detection, matching, and depth recovery
- **Recommendations**: 
  - Choose middle frames (e.g., 50) to avoid boundary effects at dataset start/end
  - Select scenes with rich texture for more features

#### ratio_threshold
- **Range**: 0.6 - 0.9
- **Default**: 0.75
- **Description**: Lowe's ratio test threshold for filtering ambiguous matches
- **Recommendations**:
  - 0.7: High precision, low recall
  - 0.8: Balanced
  - 0.9: Low precision, high recall

#### epipolar_threshold
- **Range**: 1.0 - 5.0 (pixels)
- **Default**: 2.0
- **Description**: Maximum y-coordinate difference for stereo image pairs
- **Recommendations**:
  - Rectified images: 1.0-2.0
  - Unrectified images: 3.0-5.0

#### min_depth / max_depth
- **Range**: > 0 (meters)
- **Default**: 0.5 / 15.0
- **Description**: Valid depth range for filtering outliers
- **Recommendations**:
  - Adjust based on specific scene
  - Too small min_depth includes close-range noise
  - Too large max_depth includes distant low-precision points

## FAQ

### Q1: How do I change which image pair to process?
**A:** Modify the `test_image_index` parameter in `config.yaml`, then re-run tasks 3-5.

### Q2: Why don't FAST and Harris produce matching results?
**A:** These detectors only detect keypoints without generating descriptors, so they cannot be used for matching. Only SIFT and ORB support feature matching.

### Q3: How can I increase the number of matches?
**A:** 
1. Increase `ratio_threshold` (e.g., from 0.75 to 0.8)
2. Increase `epipolar_threshold` (e.g., from 2.0 to 3.0)
3. Select image pairs with richer texture

### Q4: How do I view the point cloud files?
**A:** Use any of the following software:
- CloudCompare (free, recommended)
- MeshLab (free)
- Open3D (Python library)

### Q5: What if I can't run Task 2 on Windows?
**A:** 
1. Use a Linux virtual machine (Ubuntu 20.04 + ROS Noetic)
2. Obtain pre-extracted images from the instructor
3. Use WSL2 + ROS

### Q6: How do I batch process multiple image pairs?
**A:** You can write a simple script to loop through `test_image_index` values:
```python
import yaml
import subprocess

config_file = 'config.yaml'
for idx in range(0, 100, 10):  # Process every 10th frame
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify index
    config['feature_detection']['test_image_index'] = idx
    config['feature_matching']['test_image_index'] = idx
    
    # Save configuration
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run tasks
    subprocess.run(['python', '3. feature_detection.py'])
    subprocess.run(['python', '4. feature_matching.py'])
    subprocess.run(['python', '5. depth_recovery.py'])
```

## Lab Report Suggestions

Your report should include:

1. **Methodology**
   - Briefly describe the principles of each task
   - Explain the algorithms used (SIFT, ORB, etc.)

2. **Parameter Selection**
   - List the main parameter values used
   - Explain why these values were chosen

3. **Results**
   - Feature detection comparison charts
   - Feature matching visualizations
   - Depth maps and point clouds
   - Statistical data tables

4. **Analysis and Discussion**
   - Compare performance of different feature detectors
   - Analyze matching quality
   - Discuss depth recovery accuracy
   - Impact of parameter tuning

5. **Conclusions**
   - Summarize experimental findings
   - Propose improvement suggestions

## References

1. **EuRoC MAV Dataset**
   - Paper: Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016
   - Website: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. **Feature Detection**
   - SIFT: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004
   - ORB: Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011

3. **Stereo Vision**
   - Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd Edition
   - OpenCV Stereo Documentation: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

## Acknowledgments

This lab is developed based on the EuRoC MAV dataset from ETH Zurich. Thanks to the open-source community for excellent tools like OpenCV and ROS.

## License

This code is for educational purposes only.

---

**Last Updated:** 2025-10-29  
**Author:** AAE5306 Teaching Team  
**Course:** AAE5306 - Visual Technologies in Low-Altitude Economy
