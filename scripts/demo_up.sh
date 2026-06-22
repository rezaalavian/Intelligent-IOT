#!/usr/bin/env bash
# Bring up the full pipeline for a live demo, in the background.
#
# Model-loading services (inference, alerts, API, dashboard) run on Python 3.11
# because the TF+torch+lightgbm stack segfaults on 3.13. The model-free services
# (live_state) run on the project .venv. Override the 3.11 interpreter with PY311.
#
#   scripts/demo_up.sh        # start everything
#   scripts/demo_down.sh      # stop everything
#
# Then stream data:  make demo DEMO_MODE=eval   (or DEMO_MODE=wildfire)
set -euo pipefail
cd "$(dirname "$0")/.."

PY311="${PY311:-python3.11}"
VENV="$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)"
RUN=".demo_run"; mkdir -p "$RUN"
: > "$RUN/pids"

[ -f .env ] && { set -a; . ./.env; set +a; }

echo "==> Kafka + Schema Registry"
docker compose up -d zookeeper kafka schema-registry >/dev/null

echo -n "==> waiting for broker + registry"
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8081/subjects >/dev/null 2>&1 \
     && docker exec intelligent-iot-kafka-1 kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo " ready"; break
  fi
  echo -n "."; sleep 5
done

echo "==> topics + schemas"
$VENV -m infrastructure.kafka.create_topics    >/dev/null 2>&1 || true
$VENV -m infrastructure.kafka.register_schemas >/dev/null 2>&1 || true

echo "==> clean dashboard state"
rm -f data/stream/live_state.json

start() {  # name  interpreter  module...
  local name="$1"; shift; local py="$1"; shift
  "$py" -m "$@" > "$RUN/$name.log" 2>&1 &
  echo "$!" >> "$RUN/pids"
  echo "    started $name (pid $!)"
}

echo "==> consumers + services"
start inference  "$PY311" infrastructure.kafka.consumers.inference
start alerts     "$PY311" infrastructure.kafka.consumers.alerts
start live_state "$VENV"  infrastructure.kafka.consumers.live_state
start api        "$PY311" uvicorn infrastructure.deployment.app:app --port 8000
"$PY311" -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py \
  --server.port 8501 --server.headless true > "$RUN/dashboard.log" 2>&1 &
echo "$!" >> "$RUN/pids"; echo "    started dashboard (pid $!)"

sleep 6
echo
echo "Pipeline up. Dashboard: http://localhost:8501   API: http://localhost:8000/live/alerts"
echo "Stream data:  make demo DEMO_MODE=eval       (normal -> warning)"
echo "              make demo DEMO_MODE=wildfire    (critical / red)"
echo "Stop all:     scripts/demo_down.sh"
