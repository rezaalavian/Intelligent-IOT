#!/usr/bin/env bash
# Stop everything started by demo_up.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN=".demo_run"
if [ -f "$RUN/pids" ]; then
  while read -r pid; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "stopped pid $pid"
  done < "$RUN/pids"
  rm -f "$RUN/pids"
fi
# belt-and-suspenders in case the pid file was lost
pkill -f "infrastructure.kafka.consumers" 2>/dev/null || true
pkill -f "uvicorn infrastructure.deployment" 2>/dev/null || true
pkill -f "streamlit run infrastructure/deployment" 2>/dev/null || true

docker compose down >/dev/null 2>&1 && echo "stack down"
echo "demo stopped"
