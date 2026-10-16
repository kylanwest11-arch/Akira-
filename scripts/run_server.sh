#!/usr/bin/env bash
# Start Akira's API server in the background; logs to models/server.log
cd "$(dirname "$0")/.."
nohup nice -n 5 python3 -W ignore -m akira.server > models/server.log 2>&1 &
echo $! > models/server.pid
