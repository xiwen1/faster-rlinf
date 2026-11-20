#!/bin/bash
# Formulation: fetch the GR00T N1.5 checkpoint (RLinf/RLinf-Gr00t-SFT-Spatial, pinned revision) through hf-mirror with
# curl (resumable), and the Isaac-GR00T n1.5-release source. Checks the LFS sha256 of each shard.
set -uo pipefail
A=$ASSETS
REPO=RLinf/RLinf-Gr00t-SFT-Spatial; REV=73f710e70e7d571f8d828e51e0a428f5a1e0ac22
D=$A/weights/RLinf-Gr00t-SFT-Spatial
mkdir -p $D/experiment_cfg
echo "[$(date +%T)] clone Isaac-GR00T n1.5-release"
[ -d $A/src/Isaac-GR00T/.git ] || git clone -q --depth 1 -b n1.5-release https://github.com/NVIDIA/Isaac-GR00T.git $A/src/Isaac-GR00T
git -C $A/src/Isaac-GR00T log -1 --format="Isaac-GR00T %H %cd"
for f in config.json experiment_cfg/metadata.json model.safetensors.index.json .gitattributes model-00002-of-00002.safetensors model-00001-of-00002.safetensors; do
  for try in 1 2 3 4 5 6 7 8; do
    echo "[$(date +%T)] $f try $try"
    curl -sS -L --fail -C - --retry 3 --connect-timeout 30 -o $D/$f "https://hf-mirror.com/$REPO/resolve/$REV/$f" && break
    sleep 20
  done
done
cd $D
echo "e8e38e607765f47658ba2ccac8c7b311e57bafc58dd8490a143332c01e929556  model-00001-of-00002.safetensors
ce44c6277867741d906c3b25d281ea645edf78b0c5b8454d3c69e1a4ed6458eb  model-00002-of-00002.safetensors" | sha256sum -c && echo SHARDS_OK
ls -la $D $D/experiment_cfg
echo "[$(date +%T)] done"
