#!/usr/bin/env bash
# Start the unsupervised sweep in the background on a Lightning Studio.
set -euo pipefail

SWEEP_ARGS="${SWEEP_ARGS:?SWEEP_ARGS must be set}"
LOG="${REMOTE_LOG:-unsup_sweep.log}"
DONE="${REMOTE_DONE:-unsup_sweep.done}"
PIDFILE="${REMOTE_PID:-unsup_sweep.pid}"

WORK="$(
python -c "
from pathlib import Path
roots = [Path('.').resolve(), Path.home(), Path('/teamspace/studios/this_studio')]
for root in roots:
    for p in root.rglob('run_experiments.py'):
        try:
            txt = p.read_text(errors='ignore')
        except Exception:
            continue
        if 'Unsupervised HP search' in txt:
            print(p.parent)
            raise SystemExit
raise SystemExit('new run_experiments.py not found')
"
)"
echo "Using WORK=$WORK"
cd "$WORK"

if [ -f "$PIDFILE" ]; then
  old="$(cat "$PIDFILE" || true)"
  if [ -n "${old}" ] && kill -0 "${old}" 2>/dev/null; then
    echo "Killing prior sweep pid ${old}"
    kill "${old}" || true
    sleep 2
    kill -9 "${old}" 2>/dev/null || true
  fi
fi
pkill -f 'python run_experiments.py' 2>/dev/null || true
rm -f "$DONE" "$LOG"

nohup bash -lc "python -u run_experiments.py ${SWEEP_ARGS}; echo \$? > ${DONE}" >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started pid=$(cat "$PIDFILE")"
sleep 2
tail -n 40 "$LOG" || true
