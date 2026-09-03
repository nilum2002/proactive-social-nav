# Vendored: Slamtec rplidar_ros

Upstream : https://github.com/Slamtec/rplidar_ros
Branch   : ros2
Commit   : 24cc9b6dea97e045bda1408eaa867ce730fd3fc3 (2025-04-27)
Vendored : 2026-09-03
License  : BSD-2-Clause (see LICENSE)

Checked in without `.git` by request. That means `git log` here tells you
nothing about upstream, so the commit above is the only record of what version
this is -- update it if you re-vendor.

Fetched as a tarball rather than cloned because this machine's DNS was not
resolving github.com at the time (systemd-resolved was pointed at an IPv6
server that did not answer); the result is identical to a clone with .git
removed.

To update:
    curl -L https://codeload.github.com/Slamtec/rplidar_ros/tar.gz/refs/heads/ros2 \
        | tar xz && rm -rf rplidar_ros && mv rplidar_ros-ros2 rplidar_ros
