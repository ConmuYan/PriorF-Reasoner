#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${SESSION_NAME:-priorf_reasoner_full}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-$ROOT_DIR/priorf_reasoner_slm/scripts/run_full_pipeline.sh}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/outputs/tmux_logs}"
mkdir -p "$LOG_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed or not on PATH"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION_NAME' already exists"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
    "cd '$ROOT_DIR' && bash '$PIPELINE_SCRIPT' 2>&1 | tee '$LOG_ROOT/${SESSION_NAME}.log'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Log file: $LOG_ROOT/${SESSION_NAME}.log"
