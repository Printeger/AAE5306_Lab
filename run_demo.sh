#!/bin/bash
# Quick Run Script for AAE5306 Stereo Vision Lab
# This script helps students quickly test the package

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=================================================="
echo "AAE5306 Stereo Vision Lab - Quick Run"
echo "=================================================="
echo ""

# Check if workspace is sourced
if ! command -v rospack &> /dev/null; then
    echo -e "${RED}ROS commands not found${NC}"
    echo "Please source your workspace first:"
    echo "  source ~/aae5306_ws/devel/setup.bash"
    exit 1
fi

if ! rospack find aae5306_stereo_vision > /dev/null 2>&1; then
    echo -e "${RED}Package not found${NC}"
    echo "Please run setup.sh first or source your workspace"
    exit 1
fi

# Function to show menu
show_menu() {
    echo ""
    echo -e "${BLUE}Select an option:${NC}"
    echo "1) Complete Pipeline (Feature Detection + Matching + Depth)"
    echo "2) Feature Detection Only (SIFT)"
    echo "3) Feature Detection Only (ORB)"
    echo "4) Feature Matching Only"
    echo "5) Custom - Complete Pipeline with ORB"
    echo "6) Custom - Adjust matching parameters"
    echo "7) Help - Show topics and commands"
    echo "0) Exit"
    echo ""
}

# Function to run complete pipeline
run_complete_pipeline() {
    echo -e "${GREEN}Launching complete stereo vision pipeline...${NC}"
    echo ""
    echo "This will start:"
    echo "  - Feature detection on both cameras (SIFT)"
    echo "  - Feature matching between stereo pairs"
    echo "  - Depth recovery and point cloud generation"
    echo "  - RViz visualization"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag --clock"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision stereo_pipeline.launch
}

# Function to run feature detection with SIFT
run_feature_detection_sift() {
    echo -e "${GREEN}Launching feature detection (SIFT)...${NC}"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag"
    echo ""
    echo "View features:"
    echo "  rosrun image_view image_view image:=/feature_detection_cam0/features_image"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision feature_detection.launch detector_type:=sift
}

# Function to run feature detection with ORB
run_feature_detection_orb() {
    echo -e "${GREEN}Launching feature detection (ORB)...${NC}"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision feature_detection.launch detector_type:=orb
}

# Function to run feature matching
run_feature_matching() {
    echo -e "${GREEN}Launching feature matching...${NC}"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag"
    echo ""
    echo "View matches:"
    echo "  rosrun image_view image_view image:=/feature_matching/matches_image"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision feature_matching.launch
}

# Function to run complete pipeline with ORB
run_orb_pipeline() {
    echo -e "${GREEN}Launching complete pipeline with ORB...${NC}"
    echo ""
    echo "ORB is faster but may produce fewer features than SIFT"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag --clock"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision stereo_pipeline.launch detector_type:=orb
}

# Function to run with custom parameters
run_custom_matching() {
    echo -e "${GREEN}Launching with custom matching parameters...${NC}"
    echo ""
    echo "Default parameters:"
    echo "  ratio_threshold: 0.75"
    echo "  epipolar_threshold: 2.0"
    echo ""
    read -p "Enter ratio_threshold (0.6-0.9): " ratio
    read -p "Enter epipolar_threshold (1.0-5.0): " epipolar
    
    ratio=${ratio:-0.75}
    epipolar=${epipolar:-2.0}
    
    echo ""
    echo "Using: ratio_threshold=$ratio, epipolar_threshold=$epipolar"
    echo ""
    echo -e "${YELLOW}After launch, play the bag file in another terminal:${NC}"
    echo "  rosbag play ~/aae5306_data/MH_01_easy.bag --clock"
    echo ""
    read -p "Press Enter to continue..."
    roslaunch aae5306_stereo_vision stereo_pipeline.launch \
        ratio_threshold:=$ratio \
        epipolar_threshold:=$epipolar
}

# Function to show help
show_help() {
    echo -e "${BLUE}Useful ROS Commands:${NC}"
    echo ""
    echo "View all topics:"
    echo "  rostopic list"
    echo ""
    echo "Monitor topic frequency:"
    echo "  rostopic hz /stereo_vision/pointcloud"
    echo ""
    echo "View topic data:"
    echo "  rostopic echo /stereo_vision/match_stats"
    echo ""
    echo "View images:"
    echo "  rosrun image_view image_view image:=/stereo_vision/matches_image"
    echo ""
    echo "Plot data:"
    echo "  rqt_plot /stereo_vision/match_stats/num_final_matches"
    echo ""
    echo "View node graph:"
    echo "  rosrun rqt_graph rqt_graph"
    echo ""
    echo -e "${BLUE}Main Output Topics:${NC}"
    echo "  /stereo_vision/cam0/features_image  - Left camera features"
    echo "  /stereo_vision/matches_image        - Feature matches"
    echo "  /stereo_vision/pointcloud           - 3D point cloud"
    echo "  /stereo_vision/depth_image          - Depth visualization"
    echo "  /stereo_vision/match_stats          - Matching statistics"
    echo "  /stereo_vision/depth_stats          - Depth statistics"
    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [0-7]: " choice
    
    case $choice in
        1)
            run_complete_pipeline
            ;;
        2)
            run_feature_detection_sift
            ;;
        3)
            run_feature_detection_orb
            ;;
        4)
            run_feature_matching
            ;;
        5)
            run_orb_pipeline
            ;;
        6)
            run_custom_matching
            ;;
        7)
            show_help
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
done
