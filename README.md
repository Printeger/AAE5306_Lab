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
