#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run with sudo:"
    echo "  sudo ./setup_udev.sh"
    exit 1
fi

echo "Setting up udev rules for LD19 LiDAR and Kobuki Base..."

mkdir -p /etc/udev/rules.d

cat << 'EOF' > /etc/udev/rules.d/99-robot-devices.rules
# LD19 LiDAR (Silicon Labs CP2102)
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", SYMLINK+="ldlidar"

# Kobuki Robot Base (Yujin Robot FT232)
KERNEL=="ttyUSB*", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE="0666", SYMLINK+="kobuki"
EOF

udevadm control --reload-rules
udevadm trigger

if [ -n "$SUDO_USER" ]; then
    usermod -aG dialout "$SUDO_USER"
    echo "Added $SUDO_USER to dialout group."
fi

echo "Done! Verified symlinks:"
ls -l /dev/ldlidar /dev/kobuki 2>/dev/null || true
