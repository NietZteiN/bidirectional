#!/usr/bin/env bash
# Hold the runner at its cap without a human in the loop.
#
# WHY. Measured over the 64 h to 2026-09-14 this project achieved 0.89 GPU-hours per wall-hour,
# because work went out in bursts and the queue then drained while nobody was looking. The
# runner fixes that only if something calls it; nothing did. One pass every 15 min costs a few
# seconds of login-node CPU and keeps all three slots warm.
#
# It is safe to run unattended: the runner is idempotent (it recomputes from disk), caps itself
# at --max-inflight, and since 0486065 refuses to queue a cell that is already in flight.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/topup.log
ARGS=("$@")
END=$(( $(date +%s) + ${TOPUP_HOURS:-12} * 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  {
    echo "--- $(date -Is) ---"
    source scripts/env.sh >/dev/null 2>&1
    # TWO PASSES, PRIORITISED. The withdrawn-arm sweep goes first because its results are
    # currently unusable; whatever slots it leaves are filled by the normal plan. Running only
    # the sweep would idle every slot the moment it finished.
    if [ ${#ARGS[@]} -gt 0 ]; then
      python scripts/95_runner.py "${ARGS[@]}" 2>&1 | grep -vE "^    (PENDING|RUNNING)"
    fi
    python scripts/95_runner.py 2>&1 | grep -vE "^    (PENDING|RUNNING)"
  } >> "$LOG" 2>&1
  sleep 900
done
echo "topup loop finished at $(date -Is)" >> "$LOG"
