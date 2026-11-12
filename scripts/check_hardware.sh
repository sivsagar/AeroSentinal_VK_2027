#!/usr/bin/env bash
# =============================================================================
# AeroSentinel — Hardware Compatibility Check
# =============================================================================
# Checks OS, Python, RAM, RPi device tree, cameras, RealSense, Python packages,
# disk space, and CPU temperature.
#
# Output: coloured PASS / WARN / FAIL per check.
# Exit code: 0 if all CRITICAL checks pass, 1 otherwise.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Colour

PASS="${GREEN}[PASS]${NC}"
WARN="${YELLOW}[WARN]${NC}"
FAIL="${RED}[FAIL]${NC}"
INFO="${BLUE}[INFO]${NC}"

_critical_failures=0

pass()  { echo -e "${PASS} $*"; }
warn()  { echo -e "${WARN} $*"; }
fail()  { echo -e "${FAIL} $*"; (( _critical_failures++ )); }
info()  { echo -e "${INFO} $*"; }
header(){ echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---------------------------------------------------------------------------
# 1. OS and Architecture
# ---------------------------------------------------------------------------
header "OS & Architecture"

ARCH=$(uname -m)
OS=$(uname -s)
info "OS: ${OS}  Arch: ${ARCH}"

case "${ARCH}" in
  aarch64|arm64)
    pass "Architecture is ${ARCH} — Raspberry Pi / ARM64 target." ;;
  x86_64|amd64)
    warn "Architecture is ${ARCH} — development machine (x86). Not suitable for production deployment on RPi." ;;
  *)
    fail "Unsupported architecture: ${ARCH}" ;;
esac

# ---------------------------------------------------------------------------
# 2. Python 3 version
# ---------------------------------------------------------------------------
header "Python Version"

if ! command -v python3 &>/dev/null; then
  fail "python3 not found in PATH."
else
  PY_VER=$(python3 -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
  PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
  PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
  info "Python version: ${PY_VER}"
  if [[ "${PY_MAJOR}" -ge 3 && "${PY_MINOR}" -ge 9 ]]; then
    pass "Python ${PY_VER} meets the minimum requirement (3.9+)."
  else
    fail "Python ${PY_VER} is too old — 3.9 or later required."
  fi
fi

# ---------------------------------------------------------------------------
# 3. Available RAM
# ---------------------------------------------------------------------------
header "Available RAM"

if command -v free &>/dev/null; then
  TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
  AVAIL_RAM_MB=$(free -m | awk '/^Mem:/{print $7}')
  info "Total RAM: ${TOTAL_RAM_MB} MB | Available: ${AVAIL_RAM_MB} MB"
  if [[ "${TOTAL_RAM_MB}" -ge 2048 ]]; then
    pass "Total RAM ${TOTAL_RAM_MB} MB >= 2 GB."
  else
    warn "Total RAM ${TOTAL_RAM_MB} MB < 2 GB — may cause OOM during inference."
  fi
else
  warn "Cannot determine RAM (free not available)."
fi

# ---------------------------------------------------------------------------
# 4. Raspberry Pi device tree
# ---------------------------------------------------------------------------
header "Raspberry Pi Detection"

RPI_MODEL_FILE="/proc/device-tree/model"
if [[ -f "${RPI_MODEL_FILE}" ]]; then
  RPI_MODEL=$(tr -d '\0' < "${RPI_MODEL_FILE}")
  pass "Raspberry Pi detected: ${RPI_MODEL}"
  IS_RPI=true
else
  warn "Not running on a Raspberry Pi (/proc/device-tree/model absent). Some checks will be skipped."
  IS_RPI=false
fi

# ---------------------------------------------------------------------------
# 5. Camera interface
# ---------------------------------------------------------------------------
header "Camera Interface"

VIDEO_DEVS=$(ls /dev/video* 2>/dev/null || true)
if [[ -n "${VIDEO_DEVS}" ]]; then
  pass "Video devices found: $(echo "${VIDEO_DEVS}" | tr '\n' ' ')"
else
  warn "No /dev/video* devices found."
fi

if command -v libcamera-hello &>/dev/null; then
  info "libcamera-hello available — checking cameras…"
  LIBCAM_OUT=$(libcamera-hello --list-cameras 2>&1 || true)
  if echo "${LIBCAM_OUT}" | grep -qi "available"; then
    pass "libcamera detected at least one camera."
  else
    warn "libcamera-hello found no cameras: ${LIBCAM_OUT}"
  fi
else
  warn "libcamera-hello not installed — cannot verify Pi Camera."
fi

# ---------------------------------------------------------------------------
# 6. Intel RealSense USB device
# ---------------------------------------------------------------------------
header "Intel RealSense D435i"

if command -v lsusb &>/dev/null; then
  # Intel USB Vendor ID: 8086
  REALSENSE_USB=$(lsusb | grep -i "8086" || true)
  if [[ -n "${REALSENSE_USB}" ]]; then
    pass "Intel USB device found: ${REALSENSE_USB}"
  else
    warn "No Intel (8086) USB device found — RealSense may not be connected."
  fi
else
  warn "lsusb not available — skipping RealSense USB check."
fi

# ---------------------------------------------------------------------------
# 7. Python package imports
# ---------------------------------------------------------------------------
header "Python Package Imports"

check_import() {
  local pkg="$1"
  local critical="${2:-false}"
  if python3 -c "import ${pkg}" &>/dev/null 2>&1; then
    pass "import ${pkg}"
  else
    if [[ "${critical}" == "true" ]]; then
      fail "import ${pkg} — CRITICAL package missing."
    else
      warn "import ${pkg} — optional package not installed."
    fi
  fi
}

check_import "cv2"           "true"
check_import "yaml"          "true"
check_import "psutil"        "false"
check_import "numpy"         "true"
check_import "pyrealsense2"  "false"
check_import "picamera2"     "false"

# ---------------------------------------------------------------------------
# 8. Disk space
# ---------------------------------------------------------------------------
header "Disk Space"

DISK_AVAIL_MB=$(df -m . | awk 'NR==2{print $4}')
info "Available disk space: ${DISK_AVAIL_MB} MB"
if [[ "${DISK_AVAIL_MB}" -ge 2048 ]]; then
  pass "Disk space ${DISK_AVAIL_MB} MB >= 2 GB."
else
  warn "Disk space ${DISK_AVAIL_MB} MB < 2 GB — may run out during data collection."
fi

# ---------------------------------------------------------------------------
# 9. CPU Temperature (RPi only)
# ---------------------------------------------------------------------------
header "CPU Temperature"

THERMAL_FILE="/sys/class/thermal/thermal_zone0/temp"
if [[ "${IS_RPI}" == "true" && -f "${THERMAL_FILE}" ]]; then
  TEMP_RAW=$(cat "${THERMAL_FILE}")
  TEMP_C=$(echo "scale=1; ${TEMP_RAW}/1000" | bc 2>/dev/null || echo "N/A")
  info "CPU temperature: ${TEMP_C} °C"
  TEMP_INT=$((TEMP_RAW / 1000))
  if [[ "${TEMP_INT}" -lt 75 ]]; then
    pass "CPU temperature ${TEMP_C}°C is within safe range."
  elif [[ "${TEMP_INT}" -lt 85 ]]; then
    warn "CPU temperature ${TEMP_C}°C is elevated — check cooling."
  else
    fail "CPU temperature ${TEMP_C}°C is CRITICAL — throttling expected."
  fi
else
  info "CPU temperature check skipped (not on RPi or thermal file absent)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Summary"

if [[ "${_critical_failures}" -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}All critical checks passed.${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}${_critical_failures} critical check(s) FAILED. Review the output above.${NC}"
  exit 1
fi
