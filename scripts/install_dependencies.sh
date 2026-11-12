#!/usr/bin/env bash
# =============================================================================
# AeroSentinel — Dependency Installation Script
# Raspberry Pi 4 Bookworm 64-bit (aarch64)
# =============================================================================
# Installs system packages, creates a venv, installs Python packages,
# handles RealSense wheel availability, and runs the hardware check at the end.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

VENV_DIR="${VENV_DIR:-/opt/aerosentinel/venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"

# ---------------------------------------------------------------------------
# 1. Architecture check
# ---------------------------------------------------------------------------
info "Checking system architecture…"
ARCH=$(uname -m)
info "Architecture: ${ARCH}"

if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
  success "Running on ARM64 — Raspberry Pi target confirmed."
  IS_ARM64=true
else
  warn "Not running on ARM64 (detected: ${ARCH}). Some packages (e.g. pyrealsense2 wheel) may not be available."
  IS_ARM64=false
fi

# ---------------------------------------------------------------------------
# 2. Python version check
# ---------------------------------------------------------------------------
info "Checking Python version…"
if ! command -v python3 &>/dev/null; then
  die "python3 not found. Please install Python 3.9 or later."
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
PY_VER="${PY_MAJOR}.${PY_MINOR}"
info "Python version: ${PY_VER}"

if [[ "${PY_MAJOR}" -lt 3 || ( "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 9 ) ]]; then
  die "Python 3.9+ required. Found ${PY_VER}."
fi
success "Python ${PY_VER} OK."

# ---------------------------------------------------------------------------
# 3. System packages
# ---------------------------------------------------------------------------
info "Installing system packages via apt…"

SYSTEM_PACKAGES=(
  python3-dev
  python3-pip
  python3-venv
  libopencv-dev
  python3-opencv
  v4l-utils
  libusb-1.0-0-dev
  libglib2.0-dev
  libcap-dev
  build-essential
  cmake
  git
  curl
  wget
  bc
)

if command -v apt-get &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends "${SYSTEM_PACKAGES[@]}"
  success "System packages installed."
else
  warn "apt-get not available — skipping system package installation."
fi

# ---------------------------------------------------------------------------
# 4. picamera2 (apt-preferred on RPi)
# ---------------------------------------------------------------------------
if [[ "${IS_ARM64}" == "true" ]]; then
  info "Installing picamera2 via apt (recommended on RPi)…"
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y --no-install-recommends python3-picamera2 || \
      warn "picamera2 not found in apt — will try pip later."
    success "picamera2 installed via apt."
  fi
fi

# ---------------------------------------------------------------------------
# 5. Create / activate virtual environment
# ---------------------------------------------------------------------------
info "Setting up virtual environment at ${VENV_DIR}…"

if [[ ! -d "${VENV_DIR}" ]]; then
  sudo mkdir -p "$(dirname "${VENV_DIR}")"
  sudo python3 -m venv "${VENV_DIR}" --system-site-packages
  success "Virtual environment created."
else
  info "Virtual environment already exists at ${VENV_DIR}."
fi

# Activate venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
success "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip setuptools wheel
success "pip upgraded."

# ---------------------------------------------------------------------------
# 6. pip packages from requirements.txt
# ---------------------------------------------------------------------------
if [[ -f "${REQUIREMENTS}" ]]; then
  info "Installing Python packages from ${REQUIREMENTS}…"
  pip install -r "${REQUIREMENTS}"
  success "Python packages installed."
else
  warn "requirements.txt not found at ${REQUIREMENTS} — skipping pip install."
fi

# ---------------------------------------------------------------------------
# 7. pyrealsense2
# ---------------------------------------------------------------------------
info "Checking pyrealsense2 availability…"

if python3 -c "import pyrealsense2" &>/dev/null 2>&1; then
  success "pyrealsense2 is already importable."
elif [[ "${IS_ARM64}" == "true" ]]; then
  # Try known community wheels for RPi 4 Bookworm
  WHEEL_URL="https://github.com/IntelRealSense/librealsense/releases"
  warn "pyrealsense2 wheel for ARM64 may not be on PyPI."
  echo ""
  echo -e "${YELLOW}${BOLD}--- pyrealsense2 source-build instructions (ARM64) ---${NC}"
  echo ""
  echo "  Option A: Check the librealsense GitHub releases for a pre-built ARM64 wheel:"
  echo "    ${WHEEL_URL}"
  echo "    pip install <wheel_file>.whl"
  echo ""
  echo "  Option B: Build from source:"
  echo "    sudo apt-get install -y libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev"
  echo "    sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev"
  echo "    git clone https://github.com/IntelRealSense/librealsense.git"
  echo "    cd librealsense"
  echo "    mkdir build && cd build"
  echo "    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=\$(which python3) \\"
  echo "             -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF"
  echo "    make -j4"
  echo "    sudo make install"
  echo "    # Add /usr/local/lib to PYTHONPATH or copy the .so to site-packages"
  echo ""
  echo "  Option C: Use the community script scripts/setup_realsense.sh"
  echo ""
  echo -e "${YELLOW}${BOLD}------------------------------------------------------${NC}"
else
  pip install pyrealsense2 || warn "pyrealsense2 pip install failed — see above for ARM64 build instructions."
fi

# ---------------------------------------------------------------------------
# 8. Run hardware check
# ---------------------------------------------------------------------------
CHECK_SCRIPT="${SCRIPT_DIR}/check_hardware.sh"
if [[ -f "${CHECK_SCRIPT}" ]]; then
  info "Running hardware check…"
  bash "${CHECK_SCRIPT}" || warn "Hardware check reported issues — review output above."
else
  warn "check_hardware.sh not found at ${CHECK_SCRIPT} — skipping."
fi

echo ""
echo -e "${GREEN}${BOLD}Installation complete.${NC}"
echo "To activate the venv in future sessions:"
echo "  source ${VENV_DIR}/bin/activate"
