#!/bin/bash
# Run a command that sees exactly one GPU, as the judges expect: before every GPU run, bench/bench.py waits until
# every GPU that nvidia-smi lists is idle, and it pins CPUs to the NUMA node of nvidia-smi's GPU 0. On a host with
# several GPUs, CUDA_VISIBLE_DEVICES is not enough. This wrapper uses bubblewrap (https://github.com/containers/bubblewrap)
# the way the NVIDIA container runtime exposes one GPU: the host filesystem as it is, a fresh /dev with the NVIDIA
# control devices and /dev/nvidia<N> only, and /proc/driver/nvidia/gpus reduced to that GPU. Inside, it is GPU 0.
# A container with one GPU (e.g. docker run --gpus '"device=N"') or a single-GPU machine needs no wrapper.
#   one_gpu.sh <host GPU index> <command ...>        BWRAP: the bubblewrap binary (default: bwrap)
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: one_gpu.sh <host GPU index> <command ...>" >&2; exit 2; }
G=$1; shift
[ -e "/dev/nvidia$G" ] || { echo "one_gpu.sh: /dev/nvidia$G does not exist" >&2; exit 2; }
args=(--die-with-parent --dev-bind / / --dev /dev)
for d in nvidiactl nvidia-uvm nvidia-uvm-tools nvidia-modeset nvidia-caps; do
  [ -e "/dev/$d" ] && args+=(--dev-bind "/dev/$d" "/dev/$d")
done
args+=(--dev-bind "/dev/nvidia$G" "/dev/nvidia$G")
if [ -d /proc/driver/nvidia/gpus ]; then
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$G" | tr '[:upper:]' '[:lower:]')
  bus=${bus: -12}
  args+=(--tmpfs /proc/driver/nvidia/gpus --ro-bind "/proc/driver/nvidia/gpus/$bus" "/proc/driver/nvidia/gpus/$bus")
fi
exec "${BWRAP:-bwrap}" "${args[@]}" --setenv CUDA_VISIBLE_DEVICES 0 -- "$@"
