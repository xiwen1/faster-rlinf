#!/bin/bash
# Run repeat_check.py for one model: upstream tree (ref) then candidate tree (cand), then compare.
# Usage: repeat_check.sh <pi05|oft|gr00t> <upstream_tree> <cand_tree> <out_dir> [steps]
set -u
MODEL=$1; REF=$2; CAND=$3; OUT=$4; N=${5:-24}
HERE=$(cd "$(dirname "$0")" && pwd)
A=${FRL_ASSETS:-/path/to/assets}
case $MODEL in
  pi05)  PY=$A/venv/bin/python;       WT=$A/weights/RLinf-Pi05-LIBERO-130-fullshot-SFT ;;
  oft)   PY=$A/venv-oft/bin/python;   WT=$A/weights/Openvla-oft-SFT-libero10-traj1 ;;
  gr00t) PY=$A/venv-gr00t/bin/python; WT=$A/weights/RLinf-Gr00t-SFT-Spatial ;;
esac
mkdir -p "$OUT"; export HF_HOME=$OUT/hf_home HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
for side in ref cand; do
  T=$REF; [ $side = cand ] && T=$CAND
  (cd "$OUT" && "$PY" "$HERE/repeat_check.py" --workload "$HERE/workloads/$MODEL/workload.py" --tree "$T" \
     --weights "$WT" --mode $side --steps $N --out "$OUT/$MODEL-$side.json") > "$OUT/$MODEL-$side.log" 2>&1 \
     || { echo "$MODEL $side failed:"; tail -20 "$OUT/$MODEL-$side.log"; exit 1; }
  tail -1 "$OUT/$MODEL-$side.log"
done
"$PY" "$HERE/repeat_check.py" --compare "$OUT/$MODEL-ref.json" "$OUT/$MODEL-cand.json"
