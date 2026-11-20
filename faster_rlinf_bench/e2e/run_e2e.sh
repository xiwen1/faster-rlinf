#!/bin/bash
# End-to-end RLinf training benchmark: one model preset, one code tree, N iterations, on the GPUs in
# CUDA_VISIBLE_DEVICES (as leased by a scheduler). Per-GPU load is the original 8-GPU config's per-GPU
# load (8 envs and global_batch_size/8 per GPU); everything else is the stock config.
#
# Usage: run_e2e.sh <pi05|oft|gr00t> <tree> <tag> <iterations> [extra hydra overrides...]
# Env:   FRL_ASSETS  dir with venv/ venv-oft/ venv-gr00t/ weights/  (required)
#        FRL_E2E_OUT output root (default: <this dir>/runs)
# Output: $FRL_E2E_OUT/<tag>/{meta.txt,train.log,logs/,loadavg.txt}; then run dump_tb.py on it.
set -u
MODEL=$1; TREE=$(cd "$2" && pwd); TAG=$3; NE=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
A=${FRL_ASSETS:?set FRL_ASSETS to the assets dir, see FASTER_RLINF.md}
OUT=${FRL_E2E_OUT:-$HERE/runs}
RUN=$OUT/$TAG
mkdir -p "$RUN"
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
G=${#GL[@]}
case $MODEL in
  pi05)  PYV=$A/venv;       CFG=libero_object_ppo_openpi_pi05; W=$A/weights/RLinf-Pi05-LIBERO-130-fullshot-SFT; GB=256 ;;
  oft)   PYV=$A/venv-oft;   CFG=libero_10_grpo_openvlaoft;     W=$A/weights/Openvla-oft-SFT-libero10-traj1;   GB=2048 ;;
  gr00t) PYV=$A/venv-gr00t; CFG=libero_spatial_ppo_gr00t;      W=$A/weights/RLinf-Gr00t-SFT-Spatial;           GB=128 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
PY=$PYV/bin/python
export EMBODIED_PATH=$TREE/examples/embodiment PYTHONPATH=$TREE
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} HF_HOME=$RUN/hf_home
export RAY_TMPDIR=/tmp/frle2e-$TAG RAY_AUTH_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=$RUN/inductor
# RLinf maps placement ranks to physical GPU ids: place exactly on the leased GPUs.
if [ "$G" -eq 1 ]; then PL=${GL[0]}; else PL=${GL[0]}-${GL[$((G-1))]}; [ $((GL[G-1]-GL[0]+1)) -eq "$G" ] || { echo "non-contiguous GPUs $CUDA_VISIBLE_DEVICES"; exit 3; }; fi
export PATH=$PYV/bin:$A/venv/bin:$PATH
[ -d "$A/guard-bin" ] && export PATH=$A/guard-bin:$PATH
mkdir -p "$RAY_TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
# Private Ray head on a port block derived from the first GPU id, so runs on one node do not join each other.
B=$((21000 + ${GL[0]} * 1000))
ray start --head --temp-dir="$RAY_TMPDIR" --port=$B --node-manager-port=$((B+1)) --object-manager-port=$((B+2)) \
  --dashboard-agent-listen-port=$((B+3)) --dashboard-agent-grpc-port=$((B+4)) --metrics-export-port=$((B+5)) \
  --runtime-env-agent-port=$((B+6)) --ray-client-server-port=$((B+7)) --dashboard-port=$((B+8)) --include-dashboard=false \
  --min-worker-port=$((B+100)) --max-worker-port=$((B+999)) --disable-usage-stats > "$RUN/ray_start.log" 2>&1 \
  || { echo "ray start failed"; cat "$RUN/ray_start.log"; exit 4; }
export RAY_ADDRESS=127.0.0.1:$B
(while true; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 30; done) > "$RUN/loadavg.txt" 2>&1 &
LOADPID=$!
cd "$TREE"
{
  echo "MODEL=$MODEL CFG=$CFG TREE=$TREE commit=$(git -C "$TREE" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$TREE" status --porcelain 2>/dev/null | wc -l)"
  echo "TAG=$TAG G=$G iterations=$NE extra=$* RAY_ADDRESS=$RAY_ADDRESS"
  echo "host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start=$(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "env: $(env | grep -E "^(FSDP_|PATCH_|RLINF_)" | sort | tr "\n" " ")"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$RUN/meta.txt"
T0=$(date +%s.%N)
"$PY" examples/embodiment/train_embodied_agent.py \
  --config-path "$TREE/examples/embodiment/config" --config-name $CFG \
  "cluster.component_placement={actor\\,env\\,rollout:$PL}" runner.logger.log_path="$RUN/logs" \
  runner.max_epochs=$NE runner.save_interval=-1 \
  env.train.total_num_envs=$((8*G)) actor.global_batch_size=$((GB*G)) \
  actor.model.model_path=$W rollout.model.model_path=$W "$@" > "$RUN/train.log" 2>&1
RC=$?
T1=$(date +%s.%N)
echo "rc=$RC wall_s=$(echo "$T1 - $T0" | bc) end=$(date -Is)" >> "$RUN/meta.txt"
kill $LOADPID 2>/dev/null
pkill -f "$RAY_TMPDIR/"; sleep 5; pkill -9 -f "$RAY_TMPDIR/"
rm -rf "$RAY_TMPDIR"
"$A/venv/bin/python" "$HERE/dump_tb.py" "$RUN" || true
exit $RC
