#!/bin/bash
# Formulation launcher: retry until one idle B200 is leased, then run a PREP command in the pi05-rl sandbox on it.
#   launch.sh <tag> <command ...>     (command runs from pi05-rl/ws inside the sandbox)
set -uo pipefail
W=$WORK
TAG=$1; shift
LOG=$W/pi05-rl_formulation/evidence/$TAG.launch.log
CMD=$(printf "%q " "$@")
echo "[$(date +%T)] launcher pid $$: $CMD" >> $LOG
for i in $(seq 1 600); do
  OUT=$(mktemp -p $SUBJECT/scratch)
  cd $SUBJECT && $GPU_SCHEDULER run --gpus 1 --min-free-mb 180000 --max-util 10 \
      --exclude-host host-f --name rl-prep-$TAG -- env ${LAUNCH_ENV:-PREP=1 PI05_FORMULATION=1} \
      $GPU_IDLE_CHECK ${LAUNCH_HOST:+} $( [ -n "${LAUNCH_HOST:-}" ] || echo $W/tools/pi05-rl/agent-run.sh ) "$@" > $OUT 2>&1
  rc=$?
  if grep -q "\[idle-check\]" $OUT; then grep "\[idle-check\]" $OUT | sed "s/^/[$(date +%T)] /" >> $LOG; rm -f $OUT; sleep 60; continue; fi
  if grep -q "allocated" $OUT || ! grep -q "no matching free GPU found" $OUT; then
    cat $OUT >> $LOG; rm -f $OUT; echo "[$(date +%T)] rc=$rc" >> $LOG; echo $rc > $W/pi05-rl_formulation/evidence/$TAG.rc; exit 0
  fi
  rm -f $OUT; echo "[$(date +%T)] no idle GPU yet" >> $LOG; sleep 60
done
echo 99 > $W/pi05-rl_formulation/evidence/$TAG.rc
