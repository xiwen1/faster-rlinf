#!/bin/bash
# Host-side, inside one GPU lease (after the GPU idle check): F2 baseline (PREP), freeze (PREP), selftest, then the
# acceptance runs with the judge read-only (no PREP).   f2acc.sh <acceptance jobs ...>
set -uo pipefail
W=$WORK
RUN=$W/tools/pi05-rl/agent-run.sh
echo "=== f2 baseline $(date +%T)"
env PREP=1 PI05_FORMULATION=1 $RUN python bench/bench.py baseline --out ../.prep/f1/baseline4 --repeats 2; echo "=== rc $?"
rm -rf $W/pi05-rl/.prep/f1/baseline.old; mv $W/pi05-rl/.prep/f1/baseline $W/pi05-rl/.prep/f1/baseline.old 2>/dev/null
cp -a $W/pi05-rl/.prep/f1/baseline4 $W/pi05-rl/.prep/f1/baseline
echo "=== freeze $(date +%T)"
env PREP=1 PI05_FORMULATION=1 $RUN python ../.prep/freeze.py . ../.prep/f1 || { echo FREEZE FAILED; exit 1; }
echo "=== selftest $(date +%T)"
$RUN python -m numerics selftest --ref task/records/ref --variant ../.prep/f1/b/records --thresholds task/thresholds.json --out ../.prep/selftest.json; echo "=== rc $?"
echo "=== acceptance $(date +%T)"
$RUN bash ../.prep/acc.sh ../.prep/acc "$@"
echo "=== all done $(date +%T)"
