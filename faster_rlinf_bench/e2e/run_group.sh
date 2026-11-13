#!/bin/bash
# Run several arms of one model at the same time, one GPU per arm, on the GPUs of one lease, so all
# arms see the same host CPU load. The first arm is the baseline of the summaries.
#
# Usage: run_group.sh <pi05|oft|gr00t> <iterations> <prefix> <arm> [<arm> ...]
#   arm = name:tree[:override[:override...]]
#   An override NAME=value with an upper-case NAME is exported as an environment variable for that arm
#   only (e.g. FSDP_PINNED_OFFLOAD=0); anything else is passed to Hydra (keys absent from the YAML need
#   a leading +, e.g. +env.train.reset_prefetch=false).
# Env:  FRL_SAMPLE=1  also sample GPU memory, utilization, SM clock, power, temperature and throttle reasons every 5 s (gpu.csv) and record a py-spy profile
#                     (raw collapsed stacks, 5 Hz, non-blocking) of every arm's actor process.
#       FRL_ASSETS, FRL_E2E_OUT as in run_e2e.sh.
# Output: $FRL_E2E_OUT/<prefix>-<name>/ per arm (see run_e2e.sh) and $FRL_E2E_OUT/<prefix>-group/
#         {arms.txt, gpu.csv, apps.csv, <tag>-actor.raw, summary-<name>.md}.
set -u
MODEL=$1; NE=$2; P=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
A=${FRL_ASSETS:-/path/to/assets}
OUT=${FRL_E2E_OUT:-$HERE/runs}
GD=$OUT/$P-group
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:?}"
[ ${#GL[@]} -eq $# ] || { echo "need one GPU per arm: $# arms, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exit 2; }
mkdir -p "$GD"
printf '%s\n' "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)" "$@" > "$GD/arms.txt"

spy() {  # py-spy the actor process of run tag $1 for its whole life
  local tag=$1 pid="" p
  for _ in $(seq 720); do
    for p in $(pgrep -f 'ray::EmbodiedFSDPActor'); do
      tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/frle2e-$tag" && { pid=$p; break; }
    done
    [ -n "$pid" ] && break
    sleep 5
  done
  [ -n "$pid" ] || { echo "no actor process found for $tag" > "$GD/$tag-actor.pyspy.log"; return; }
  echo "pid=$pid start=$(date -Is)" > "$GD/$tag-actor.pyspy.log"
  py-spy record --pid "$pid" --rate 5 --idle --nonblocking --format raw -o "$GD/$tag-actor.raw" >> "$GD/$tag-actor.pyspy.log" 2>&1
}

SAMPLERS=()
if [ "${FRL_SAMPLE:-0}" = 1 ]; then
  export PATH=$A/venv/bin:$PATH  # py-spy
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader \
    -i "$CUDA_VISIBLE_DEVICES" -l 5 > "$GD/gpu.csv" 2>&1 & SAMPLERS+=($!)
  # every process on the leased GPUs (to check that nothing else runs on them)
  (while true; do nvidia-smi --query-compute-apps=timestamp,gpu_bus_id,pid,used_memory,process_name \
     --format=csv,noheader; sleep 30; done) > "$GD/apps.csv" 2>&1 & SAMPLERS+=($!)
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader > "$GD/gpu-bus.csv"
fi

PIDS=(); NAMES=(); i=0
for spec in "$@"; do
  IFS=: read -ra F <<< "$spec"
  name=${F[0]}; tree=${F[1]}; envs=(); hy=()
  for o in "${F[@]:2}"; do
    if [[ $o =~ ^[A-Z_][A-Z0-9_]*= ]]; then envs+=("$o"); else hy+=("$o"); fi
  done
  env ${envs[@]+"${envs[@]}"} CUDA_VISIBLE_DEVICES=${GL[$i]} \
    "$HERE/run_e2e.sh" "$MODEL" "$tree" "$P-$name" "$NE" ${hy[@]+"${hy[@]}"} & PIDS+=($!)
  NAMES+=("$name")
  [ "${FRL_SAMPLE:-0}" = 1 ] && { spy "$P-$name" & SAMPLERS+=($!); }
  i=$((i + 1))
  sleep 20  # stagger Ray start-up
done
for k in "${!PIDS[@]}"; do wait "${PIDS[$k]}"; echo "${NAMES[$k]} rc=$? end=$(date -Is)" >> "$GD/arms.txt"; done
sleep 10
kill "${SAMPLERS[@]}" 2>/dev/null
for name in "${NAMES[@]:1}"; do
  "$A/venv/bin/python" "$HERE/analyze_e2e.py" "$OUT/$P-${NAMES[0]}" "$OUT/$P-$name" > "$GD/summary-$name.md" 2>&1
  cat "$GD/summary-$name.md"
done
