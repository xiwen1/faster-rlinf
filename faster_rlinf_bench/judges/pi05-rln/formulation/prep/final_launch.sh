#!/bin/bash
# pi05-rln final judgement. final_launch.sh check|extra
#   check: full check of the clean checkout on new observation streams (hidden offset), reference recorded afresh,
#          in the agent sandbox on one idle GPU; INVALID -> measure again (at most 8 measurements).
#   extra: formulator-side extra configurations (extra_inner.sh) on one idle GPU.
W=$WORK
F=$W/pi05-rln/.final; M=$W/pi05-rln_formulation/final
KIND=$1; LOG=$M/final_$KIND.log; n=0
OFF=$(python3 -c "import json; print(json.load(open('$M/secret.json'))['obs_seed_offset'])")
echo "[$(date +%T)] launcher pid $$ ($KIND)" >> $LOG
for i in $(seq 1 720); do
  A=$(mktemp -p $SUBJECT/scratch)
  if [ $KIND = check ]; then
    CMD=(python bench/bench.py check --tree $F/RLinf --out $F/out-$i --obs-seed-offset $OFF --fresh-reference)
  else
    CMD=(bash $F/extra_inner.sh)
  fi
  cd $SUBJECT && $GPU_SCHEDULER run --gpus 1 --min-free-mb 180000 \
      --max-util 10 --exclude-host host-f --name rln-final-$KIND -- $GPU_IDLE_CHECK \
      $W/tools/pi05-rln/agent-run.sh "${CMD[@]}" > $A 2>&1
  rc=$?
  if grep -q "\[idle-check\]" $A; then grep "\[idle-check\]" $A | sed "s/^/[$(date +%T)] /" >> $LOG; rm -f $A; sleep 60; continue; fi
  if grep -q "allocated" $A || ! grep -q "no matching free GPU found" $A; then
    cat $A >> $LOG; rm -f $A; echo "[$(date +%T)] attempt $i rc $rc" >> $LOG
    if [ $KIND = check ]; then
      n=$((n+1))
      st=$(python3 -c "import json; print(json.load(open('$F/out-$i/check.json'))['status'])" 2>/dev/null)
      if [ "$st" = INVALID ] && [ $n -lt 8 ]; then echo "[$(date +%T)] INVALID, measuring again" >> $LOG; sleep 60; continue; fi
    fi
    echo "$i" > $M/${KIND}_done; exit 0
  fi
  rm -f $A; echo "[$(date +%T)] no idle GPU yet" >> $LOG; sleep 60
done
echo 99 > $M/${KIND}_done
