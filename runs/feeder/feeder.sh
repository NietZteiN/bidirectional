#!/usr/bin/env bash
# Detached hold-node startup enters a persistent supervisor. No ten-day expiry.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/feeder/.launcher.lock
flock -n 9 || { echo "pipeline launcher already running"; exit 0; }
trap 'exit 0' TERM INT
while [ ! -e runs/feeder/STOP ]; do
  ( source scripts/env.sh && exec python runs/feeder/supervise.py ) >>runs/feeder/launcher.log 2>&1
  [ -e runs/feeder/STOP ] && break
  sleep 10
done
