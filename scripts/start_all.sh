#!/usr/bin/env bash
# Start (or restart after a reboot) Akira's training queue and API server.
# Safe to run any time: each is started only if it isn't already running.
cd "$(dirname "$0")/.."
mkdir -p models
if ! pgrep -f "scripts/train_queu[e].py" > /dev/null; then
  nohup python3 scripts/train_queue.py >> models/queue.log 2>&1 &
  echo "training queue started"
else
  echo "training queue already running"
fi
if ! pgrep -f "[a]kira.server" > /dev/null; then
  nohup nice -n 5 python3 -W ignore -m akira.server >> models/server.log 2>&1 &
  echo "server started on :8000"
else
  echo "server already running"
fi
