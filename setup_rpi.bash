#!/usr/bin/env bash

# ROS2 JAZZY setup for Raspberry Pi 4B
echo "##############################"
echo "## Setting up Raspberry Pi  ##"
echo "##############################"

set -e

echo "--- Current Locale ---"
locale

echo "--- Updating package list & installing locales ---"
sudo apt update && sudo apt install -y locales

echo "--- Generating en_US.UTF-8 locale ---"
sudo locale-gen en_US.UTF-8

echo "--- Updating system-wide locale settings ---"
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "--- Exporting environment variable for current session ---"
export LANG=en_US.UTF-8

echo "--- Verified Settings ---"
locale

# set universe repository
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop

source /opt/ros/jazzy/setup.bash
echo "Setup colcon"
sudo apt install python3-colcon-common-extensions
echo "Starting ROS 2 Talker node for 10 seconds..."

# Runs the command and sends SIGINT (Ctrl+C) after 10 seconds
#timeout --signal=SIGINT 10s ros2 run demo_nodes_cpp talker

echo "Talker node stopped cleanly."

echo "##############################"
echo "## All set up for ROS2      ##"
echo "##############################"




# Setup git hub for lidar

echo "##############################"
echo "#     Setting up Lidar PKG   #"
echo "##############################"



cd ros2_ws/src
git clone "https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2"

echo "##############################"
echo "#   Lidar Setup Done         #"
echo "##############################"


cd ..

echo "######## Enable SSH ###########"

sudo apt install openssh-server -y
sudo systemctl enable --now ssh



sudo python3 -m pip  install --break-system-packages grpcio grpcio-tools
echo "##############################"
echo "#         All Done           #"
echo "##############################"














# set up dependencies for lidar

