#!/usr/bin/env bash
# ==============================================================================
# Intel RealSense D435i Setup for Raspberry Pi 4B (Debian/Bookworm 64-bit)
# ==============================================================================

set -eo pipefail

echo "=========================================================="
echo "  AeroSentinel Intel RealSense D435i Installer for Pi 4B   "
echo "=========================================================="

# 1. Check Root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo:"
    echo "  sudo ./scripts/setup_realsense.sh"
    exit 1
fi

# 2. Add Udev Rules
echo "[1/4] Installing RealSense udev rules..."
curl -sSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
    -o /etc/udev/rules.d/99-realsense-libusb.rules
udevadm control --reload-rules && udevadm trigger
echo "udev rules installed."

# 3. Check for pre-built pyrealsense2 wheel or librealsense packages
echo "[2/4] Checking apt repository for librealsense2..."
apt-get update -y
if apt-cache show librealsense2-dev &>/dev/null; then
    echo "Installing librealsense2 from apt..."
    apt-get install -y librealsense2-dev librealsense2-utils python3-pyrealsense2
    echo "Installed via apt."
    exit 0
fi

# 4. Source Build Instructions if package not in upstream apt
echo "[3/4] librealsense not found in distro apt. Installing build dependencies..."
apt-get install -y \
    git cmake build-essential \
    libssl-dev libusb-1.0-0-dev pkg-config \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev python3-pip

BUILD_DIR="/tmp/librealsense_build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -d "librealsense" ]; then
    echo "Cloning librealsense v2.54.2..."
    git clone --depth 1 --branch v2.54.2 https://github.com/IntelRealSense/librealsense.git
fi

cd librealsense
mkdir -p build && cd build

echo "[4/4] Configuring CMake for Raspberry Pi 4B (ARM64, Python Bindings, No OpenGL/CUDA)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=true \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_UNIT_TESTS=false

echo "Building librealsense with 4 parallel jobs (this may take 45-60 minutes on Pi 4B)..."
make -j4
make install
ldconfig

echo "Verifying pyrealsense2 import..."
python3 -c "import pyrealsense2 as rs; print('librealsense version:', rs.__version__)" || {
    echo "Warning: Python bindings built but not in default site-packages. Check /usr/local/lib/python3.*/site-packages."
}

echo "=========================================================="
echo "RealSense D435i installation complete!"
echo "=========================================================="

