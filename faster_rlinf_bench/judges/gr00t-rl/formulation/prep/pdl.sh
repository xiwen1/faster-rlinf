#!/bin/bash
# Parallel ranged download of one checkpoint file through hf-mirror (the xet bridge serves ~2 MB/s per connection;
# a single connection stalled at ~260 KB/s).   pdl.sh <file> <total size> <sha256> [parallel]
# Resumes from the existing partial file (bytes already present are kept).
set -uo pipefail
A=$ASSETS
REPO=RLinf/RLinf-Gr00t-SFT-Spatial; REV=73f710e70e7d571f8d828e51e0a428f5a1e0ac22
D=$A/weights/RLinf-Gr00t-SFT-Spatial; F=$1; TOTAL=$2; SHA=$3; P=${4:-12}
U="https://hf-mirror.com/$REPO/resolve/$REV/$F"
PARTS=$D/.parts-$(basename $F); mkdir -p $PARTS
START=$(stat -c %s $D/$F 2>/dev/null || echo 0)
CH=$((64*1024*1024))
echo "[$(date +%T)] $F: have $START of $TOTAL bytes, chunks of $CH, $P parallel"
seq $START $CH $((TOTAL-1)) > $PARTS/starts
fetch() {
  s=$1; e=$((s+CH-1)); [ $e -ge $TOTAL ] && e=$((TOTAL-1)); want=$((e-s+1)); out=$PARTS/$(printf "%012d" $s)
  for t in $(seq 1 30); do
    [ -f $out ] && [ $(stat -c %s $out) -eq $want ] && return 0
    curl -s -L --fail --connect-timeout 30 -m 900 -r $s-$e -o $out.tmp "$U" && [ $(stat -c %s $out.tmp) -eq $want ] && mv $out.tmp $out && return 0
    sleep 5
  done
  echo "chunk $s failed"; return 1
}
export -f fetch; export CH TOTAL U PARTS
xargs -P $P -I{} bash -c "fetch {}" < $PARTS/starts || { echo "[$(date +%T)] some chunks failed"; exit 1; }
for s in $(cat $PARTS/starts); do cat $PARTS/$(printf "%012d" $s) >> $D/$F; done
[ $(stat -c %s $D/$F) -eq $TOTAL ] || { echo "[$(date +%T)] size mismatch"; exit 1; }
echo "$SHA  $D/$F" | sha256sum -c && rm -rf $PARTS && echo "[$(date +%T)] $F OK"
