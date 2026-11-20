#!/bin/bash
# Finish the checkpoint: shard 1 by parallel ranges, then verify shard 2 (fetched whole by download.sh).
F=$WORK/gr00t-rl_formulation
D=$ASSETS/weights/RLinf-Gr00t-SFT-Spatial
bash $F/prep/pdl.sh model-00001-of-00002.safetensors 4999367032 e8e38e607765f47658ba2ccac8c7b311e57bafc58dd8490a143332c01e929556 16
cd $D && echo "ce44c6277867741d906c3b25d281ea645edf78b0c5b8454d3c69e1a4ed6458eb  model-00002-of-00002.safetensors" | sha256sum -c && echo SHARD2_OK
ls -la $D $D/experiment_cfg
echo "[$(date +%T)] PDL_DONE"
