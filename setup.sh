#!/bin/bash
# Quick Setup Script for AAE5306 Stereo Vision Lab
# This script helps set up the ROS package quickly

set -e

echo "=================================================="
echo "AAE5306 Stereo Vision Lab - Quick Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ROS is installed
echo -n "Checking ROS installation... "
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    echo -e "${GREEN}OK${NC}"
    source /opt/ros/noetic/setup.bash
else
    echo -e "${RED}FAILED${NC}"
    echo "ROS Noetic not found. Please install ROS Noetic first."
    exit 1
fi

# Check Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ $(echo "$PYTHON_VERSION >= 3.7" | bc) -eq 1 ]; then
    echo -e "${GREEN}OK (Python $PYTHON_VERSION)${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Python 3.7+ required, found Python $PYTHON_VERSION"
    exit 1
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-image-view \
    ros-noetic-rqt-image-view \
    ros-noetic-vision-opencv \
    ros-noetic-message-filters \
    python3-pip \
    bc

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install opencv-contrib-python numpy

# Verify OpenCV SIFT
echo -n "Checking OpenCV SIFT support... "
SIFT_CHECK=$(python3 -c "import cv2; print(hasattr(cv2, 'SIFT_create'))" 2>/dev/null)
if [ "$SIFT_CHECK" = "True" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "SIFT not available. Installing opencv-contrib-python..."
    pip3 uninstall -y opencv-python opencv-python-headless
    pip3 install opencv-contrib-python
fi

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're already in a catkin workspace
if [ -f "$SCRIPT_DIR/../CMakeLists.txt" ] && grep -q "catkin" "$SCRIPT_DIR/../CMakeLists.txt" 2>/dev/null; then
    echo ""
    echo -e "${GREEN}Already in a catkin workspace${NC}"
    WORKSPACE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
else
    # Create workspace if it doesn't exist
    WORKSPACE_DIR="$HOME/aae5306_ws"
    if [ ! -d "$WORKSPACE_DIR" ]; then
        echo ""
        echo "Creating catkin workspace at $WORKSPACE_DIR..."
        mkdir -p "$WORKSPACE_DIR/src"
        cd "$WORKSPACE_DIR"
        catkin_make
        echo -e "${GREEN}Workspace created successfully${NC}"
    else
        echo ""
        echo -e "${YELLOW}Workspace already exists at $WORKSPACE_DIR${NC}"
    fi

    # Link or copy package to workspace
    PACKAGE_NAME="aae5306_stereo_vision"
    PACKAGE_LINK="$WORKSPACE_DIR/src/$PACKAGE_NAME"

    if [ ! -e "$PACKAGE_LINK" ]; then
        echo ""
        echo "Linking package to workspace..."
        ln -s "$SCRIPT_DIR" "$PACKAGE_LINK"
        echo -e "${GREEN}Package linked successfully${NC}"
    else
        echo ""
        echo -e "${YELLOW}Package already exists in workspace${NC}"
    fi
fi

# Build workspace
echo ""
echo "Building workspace..."
cd "$WORKSPACE_DIR"
catkin_make

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Setup bashrc
BASHRC_LINE="source $WORKSPACE_DIR/devel/setup.bash"
if ! grep -q "$BASHRC_LINE" ~/.bashrc; then
    echo ""
    echo "Adding workspace to ~/.bashrc..."
    echo "# AAE5306 Workspace" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    echo -e "${GREEN}Added to ~/.bashrc${NC}"
else
    echo ""
    echo -e "${YELLOW}Workspace already in ~/.bashrc${NC}"
fi

# Source the workspace
source "$WORKSPACE_DIR/devel/setup.bash"

# Verify installation
echo ""
echo "Verifying installation..."
if rospack find aae5306_stereo_vision > /dev/null 2>&1; then
    echo -e "${GREEN}Package found successfully!${NC}"
    PACKAGE_PATH=$(rospack find aae5306_stereo_vision)
    echo "Package location: $PACKAGE_PATH"
else
    echo -e "${RED}Package not found!${NC}"
    echo "Please source the workspace manually:"
    echo "  source $WORKSPACE_DIR/devel/setup.bash"
fi

# Check for data directory
echo ""
echo "Checking for EuRoC dataset..."
DATA_DIR="$HOME/aae5306_data"
BAG_FILE="$DATA_DIR/MH_01_easy.bag"

if [ -f "$BAG_FILE" ]; then
    echo -e "${GREEN}Dataset found: $BAG_FILE${NC}"
else
    echo -e "${YELLOW}Dataset not found${NC}"
    echo "To download the EuRoC dataset, run:"
    echo "  mkdir -p $DATA_DIR"
    echo "  cd $DATA_DIR"
    echo "  wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "To use the package:"
echo "  1. Open a new terminal (or run: source ~/.bashrc)"
echo "  2. Launch the pipeline: roslaunch aae5306_stereo_vision stereo_pipeline.launch"
echo "  3. In another terminal, play the bag: rosbag play MH_01_easy.bag --clock"
echo ""
echo "For more information, see README_ROS.md"
echo ""
