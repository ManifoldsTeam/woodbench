#!/bin/bash
# Monitor training progress in watch mode
# Usage: bash monitor_progress.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting progress monitor..."
echo "Press Ctrl+C to stop"
echo ""

python "$SCRIPT_DIR/check_progress.py" --watch --verbose
