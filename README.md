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

**Last Updated:** 2025-10-29  
**Author:** AAE5306 Teaching Team  
**Course:** AAE5306 - Visual Technologies in Low-Altitude Economy
