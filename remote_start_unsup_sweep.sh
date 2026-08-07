#!/usr/bin/env bash
# Start the unsupervised sweep in the background on a Lightning Studio.
# If REMOTE_STOP_AFTER=1, stop this Studio after the sweep finishes.
set -euo pipefail

SWEEP_ARGS="${SWEEP_ARGS:?SWEEP_ARGS must be set}"
LOG="${REMOTE_LOG:-unsup_sweep.log}"
DONE="${REMOTE_DONE:-unsup_sweep.done}"
PIDFILE="${REMOTE_PID:-unsup_sweep.pid}"
REMOTE_STOP_AFTER="${REMOTE_STOP_AFTER:-0}"

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
pkill -f 'train_pca_baseline.py' 2>/dev/null || true
pkill -f 'train_lv_vsbm.py' 2>/dev/null || true
pkill -f 'train_lv_e.py' 2>/dev/null || true
pkill -f 'gnn_vsbm.py' 2>/dev/null || true
pkill -f 'gnn_e_vsbm.py' 2>/dev/null || true
rm -f "$DONE" "$LOG"
rm -f hp_* final_* 2>/dev/null || true

WRAPPER="$WORK/remote_run_unsup_sweep.sh"
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
set +e
python -u run_experiments.py ${SWEEP_ARGS}
ec=\$?
set -e
echo \$ec > "$DONE"
if [ "$REMOTE_STOP_AFTER" = "1" ]; then
  echo "Stopping Studio after sweep (exit=\$ec)..."
  python "$WORK/remote_stop_studio.py" || echo "Studio stop failed; check credentials/env"
fi
exit \$ec
EOF
chmod +x "$WRAPPER"

nohup bash "$WRAPPER" >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started pid=$(cat "$PIDFILE") remote_stop_after=$REMOTE_STOP_AFTER"
sleep 3
tail -n 40 "$LOG" || true
