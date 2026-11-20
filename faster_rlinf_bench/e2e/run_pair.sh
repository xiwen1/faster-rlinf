#!/bin/bash
# Run the reference and the candidate tree at the same time on two GPUs of one node (one lease of 2
# GPUs), so both see the same host CPU load. GPU order alternates with SWAP=1.
# Usage: run_pair.sh <model> <ref_tree> <cand_tree> <tag_prefix> <iterations> [extra hydra overrides...]
set -u
MODEL=$1; REF=$2; CAND=$3; P=$4; NE=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd)
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:?}"
[ ${#GL[@]} -eq 2 ] || { echo "need exactly 2 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 2; }
GR=${GL[0]}; GC=${GL[1]}
[ "${SWAP:-0}" = 1 ] && { GR=${GL[1]}; GC=${GL[0]}; }
CUDA_VISIBLE_DEVICES=$GR "$HERE/run_e2e.sh" "$MODEL" "$REF" "$P-ref" "$NE" "$@" & PR=$!
sleep 20  # stagger Ray start-up
CUDA_VISIBLE_DEVICES=$GC "$HERE/run_e2e.sh" "$MODEL" "$CAND" "$P-cand" "$NE" "$@" & PC=$!
wait $PR; RR=$?; wait $PC; RC=$?
echo "ref rc=$RR cand rc=$RC"
"${FRL_ASSETS:?set FRL_ASSETS to the assets dir, see FASTER_RLINF.md}/venv/bin/python" \
  "$HERE/analyze_e2e.py" "${FRL_E2E_OUT:-$HERE/runs}/$P-ref" "${FRL_E2E_OUT:-$HERE/runs}/$P-cand" \
  > "${FRL_E2E_OUT:-$HERE/runs}/$P-summary.md" 2>&1
cat "${FRL_E2E_OUT:-$HERE/runs}/$P-summary.md"
