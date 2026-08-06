#!/usr/bin/env bash
# Print status of the detached unsupervised sweep.
set -euo pipefail
cd /teamspace/studios/this_studio
PIDFILE=unsup_sweep.pid
DONE=unsup_sweep.done
LOG=unsup_sweep.log
pid="$(cat "$PIDFILE" 2>/dev/null || true)"
alive=0
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
done=0
[ -f "$DONE" ] && done=1
echo "STATUS alive=$alive done=$done pid=$pid"
tail -n 25 "$LOG" 2>/dev/null || true
