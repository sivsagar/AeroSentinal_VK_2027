#!/usr/bin/env bash
# ==============================================================================
# AeroSentinel Production Daemon & Process Launcher
# ==============================================================================

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PID_FILE="/tmp/aerosentinel.pid"
LOG_DIR="$REPO_DIR/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/aerosentinel_$(date +'%Y%m%d_%H%M%S').log"

# Activate Python Virtual Environment if present
if [ -d "$REPO_DIR/.venv" ]; then
    source "$REPO_DIR/.venv/bin/activate"
elif [ -d "/opt/aerosentinel/venv" ]; then
    source "/opt/aerosentinel/venv/bin/activate"
fi

# Load Environment Variables from .env if present
if [ -f "$REPO_DIR/.env" ]; then
    export $(grep -v '^#' "$REPO_DIR/.env" | xargs)
fi

echo "=========================================================="
echo "  Starting AeroSentinel UAV Disaster S&R Platform         "
echo "  Mode: ${AEROSENTINEL_MOCK_MODE:-hardware}               "
echo "  Log File: $LOG_FILE                                     "
echo "=========================================================="

# Check hardware if not in mock mode
if [ "${AEROSENTINEL_MOCK_MODE}" != "true" ] && [ "$1" != "--mock" ]; then
    echo "Running hardware diagnostic check..."
    bash "$REPO_DIR/scripts/check_hardware.sh" || {
        echo "Hardware verification failed. Use --mock flag to run in simulation mode."
        exit 1
    }
fi

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:$PYTHONPATH"

# Launch main application
python3 -m aerosentinel.main "$@" > >(tee -a "$LOG_FILE") 2>&1 &
APP_PID=$!
echo $APP_PID > "$PID_FILE"

echo "AeroSentinel started with PID $APP_PID"
echo "Dashboard live at: http://$(hostname -I | awk '{print $1}'):8080"
echo "To follow logs: tail -f $LOG_FILE"
echo "To stop: kill $APP_PID"

