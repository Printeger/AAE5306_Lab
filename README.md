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
	│   │       ├── optical_flow_matching_node.py
	│   │       ├── video_temporal_tracking_node.py
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

## Optical Flow Video Tracking Node

The file `video_temporal_tracking_node.py` streams frames from an MP4 file and
tracks a single moving object using Shi-Tomasi corners with Lucas-Kanade optical
flow. The node publishes both the raw frames and an overlay image showing the
tracked feature trails, plus timing statistics via `MatchStats`.

### Launching the tracker

Use the provided launch file to load configuration parameters and start the
node:

```bash
roslaunch aae5306_stereo_vision video_temporal_tracking.launch \
	video_file:=/home/cooper/code/ws/AAE5306_Lab/drone_fly_downscale.mp4 \
	visualize:=true \
	loop_video:=true
```

Key runtime parameters (`rosparam` or launch arguments):

- `video_file` – path to the MP4 clip to track (defaults to the downscaled drone video).
- `frame_rate` – processing timer frequency in Hz.
- `visualize` – toggles publishing of `/stereo_vision/video_frame` and `/stereo_vision/video_temporal_tracks`.
- `loop_video` – rewinds the clip when the end is reached for continuous demos.

Published outputs:

- `/stereo_vision/video_frame` (`sensor_msgs/Image`) – original BGR frame stream.
- `/stereo_vision/video_temporal_tracks` (`sensor_msgs/Image`) – overlay with feature trails and IDs.
- `/stereo_vision/video_track_stats` (`aae5306_stereo_vision/MatchStats`) – telemetry with timing and active track counts.

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

## Q&A
1. Map Windows serial to WSL
   If you are using WSL, you need to map the serial to WSL, in this way you can connect the camera in WSL.
   1) Download and install usbipd:
      https://github.com/dorssel/usbipd-win/releases/latest
      download x64 msi, and install.
   2) List all of the USB devices connected to Windows by opening PowerShell in administrator mode and entering the following command. Once the devices are listed, select and copy the bus ID of the device you’d like to attach to WSL.
      ```
      usbipd list
      ```
   3) Before attaching the USB device, the command usbipd bind must be used to share the device, allowing it to be attached to WSL. This requires administrator privileges. Select the bus ID of the device you would like to use in WSL and run the following command. After running the command, verify that the device is shared using the command usbipd list again.
      ```
      usbipd bind --busid 4-4
      ```
   4) To attach the USB device, run the following command. (You no longer need to use an elevated administrator prompt.) Ensure that a WSL command prompt is open in order to keep the WSL 2 lightweight VM active. Note that as long as the USB device is attached to WSL, it cannot be used by Windows. Once attached to WSL, the USB device can be used by any distribution running as WSL 2. Verify that the device is attached using usbipd list. From the WSL prompt, run lsusb to verify that the USB device is listed and can be interacted with using Linux tools.
      ```
		usbipd attach --wsl --busid <busid>
      ```
   5) Open Ubuntu (or your preferred WSL command line) and list the attached USB devices using the command:
      ```
      lsusb
      ```

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

**Last Updated:** 2025-11-06  
**Author:** AAE5306 Teaching Team  
**Course:** AAE5306 - Visual Technologies in Low-Altitude Economy
