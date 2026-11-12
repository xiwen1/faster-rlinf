#!/bin/bash
# Run faster_rlinf_bench/env_bench.py against a given RLinf tree with the LIBERO runtime env.
# Usage: env_run.sh <tree> <python> <env_bench.py args...>
set -u
TREE=$1; PY=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$TREE EMBODIED_PATH=$TREE/examples/embodiment
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
cd "$TREE"
exec "$PY" "$HERE/env_bench.py" "$@"
