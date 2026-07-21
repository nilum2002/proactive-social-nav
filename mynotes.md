# 2D LiDAR Person Detection Notes

This document contains instructions to set up the environment, build the ROS 2 workspace, and run the LDROBOT D500 LiDAR along with the DR-SPAAM person detector.

---

## Environment Setup
Create the virtual environment and install dependencies:
```bash
# 1. Create venv
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install packages
pip install -r req.txt
pip install -e .
```

---

## Building the ROS 2 Workspace (Jazzy)
To compile the ROS 2 workspace (`dr_spaam_ros2` and `ldlidar_stl_ros2`), make sure the virtual environment is **deactivated** so CMake uses the default system python:

```bash
# 1. Deactivate venv (if active)
deactivate

# 2. Go to the workspace directory
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws

# 3. Clean up old build caches (if any)
rm -rf build/ install/ log/

# 4. Compile with the system Python path
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

---

## Running the Nodes (Step-by-Step)

### Terminal 1: LDROBOT D500 LiDAR Driver
Ensure the LiDAR is plugged in (defaults to `/dev/ttyUSB0`):
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Go to workspace and source it
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws
source install/setup.bash

# Launch driver
ros2 launch ldlidar_stl_ros2 ld19.launch.py
```

### Terminal 2: DR-SPAAM Person Detector
```bash
# 1. Source ROS 2
source /opt/ros/jazzy/setup.bash

# 2. Activate your virtual environment
source /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/bin/activate

# 3. Export the PYTHONPATH to load the local ML module and its venv dependencies
export PYTHONPATH=$PYTHONPATH:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/dr_spaam:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/lib/python3.12/site-packages

# 4. Source the built workspace
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws
source install/setup.bash

# 5. Launch the detector node
ros2 launch dr_spaam_ros2 dr_spaam_ros2.launch.py
```

### Terminal 3: Visualization (RViz2)
```bash
# Source ROS 2 and open visualizer
source /opt/ros/jazzy/setup.bash
rviz2
```

**Inside RViz2 Configuration**:
* **Fixed Frame**: Set to `base_laser`
* Click **Add** -> **By topic**:
  * `/scan` -> Select **LaserScan**
  * `/dr_spaam_ros2_node/rviz_marker` -> Select **Marker**
