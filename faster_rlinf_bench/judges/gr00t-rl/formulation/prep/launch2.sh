#!/bin/bash
# Formulation launcher, second version (after F1's disturbed baseline): launch.sh with --min-free-mb 181500 (as oft-rl:
# 180000 kept leasing GPUs that already held 1-2 GB) and host selection from $LAUNCH_HOSTARGS (default:
# exclude host-f, host-e = F1's disturbed host, host-d = all GPUs busy).
#   [LAUNCH_HOSTARGS="--host host-b"] launch2.sh <tag> <command ...>   (command runs from gr00t-rl/ws)
set -uo pipefail
W=$WORK
TAG=$1; shift
LOG=$W/gr00t-rl_formulation/evidence/$TAG.launch.log
HOSTARGS=${LAUNCH_HOSTARGS:---exclude-host host-f --exclude-host host-e --exclude-host host-d}
CMD=$(printf "%q " "$@")
echo "[$(date +%T)] launcher pid $$ ($HOSTARGS): $CMD" >> $LOG
for i in $(seq 1 600); do
  OUT=$(mktemp -p $SUBJECT/scratch)
  cd $SUBJECT && $GPU_SCHEDULER run --gpus 1 --min-free-mb 181500 --max-util 10 \
      $HOSTARGS --name g00-prep-$TAG -- env ${LAUNCH_ENV:-PREP=1 GR00T_FORMULATION=1} \
      $GPU_IDLE_CHECK $W/tools/gr00t-rl/agent-run.sh "$@" > $OUT 2>&1
  rc=$?
  if grep -q "\[idle-check\]" $OUT; then grep "\[idle-check\]" $OUT | sed "s/^/[$(date +%T)] /" >> $LOG; rm -f $OUT; sleep 60; continue; fi
  if grep -q "allocated" $OUT || ! grep -q "no matching free GPU found" $OUT; then
    cat $OUT >> $LOG; rm -f $OUT; echo "[$(date +%T)] rc=$rc" >> $LOG; echo $rc > $W/gr00t-rl_formulation/evidence/$TAG.rc; exit 0
  fi
  rm -f $OUT; echo "[$(date +%T)] no idle GPU yet" >> $LOG; sleep 60
done
echo 99 > $W/gr00t-rl_formulation/evidence/$TAG.rc
