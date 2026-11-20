#!/bin/bash
# Acceptance: the frozen judge on the acceptance candidates, as the agent would run it (no PREP: the judge is
# read-only). Inside the sandbox on one leased GPU:  acc.sh <out root> <candidate>:<quick|full>[:<hidden offset>] ...
set -uo pipefail
O=$1; shift
mkdir -p $O
for job in "$@"; do
  IFS=: read -r name mode offset <<< "$job"
  flags=()
  [ "$mode" = quick ] && flags+=(--quick)
  [ -n "${offset:-}" ] && flags+=(--obs-seed-offset $offset --fresh-reference)
  tag=$name-$mode${offset:+-final}
  [ -e $O/$tag ] && mv $O/$tag $O/$tag.prev-$(date +%H%M%S)
  echo "=== $tag $(date +%T)"
  python bench/bench.py check --tree $(pwd)/../.prep/accept/$name --out $O/$tag "${flags[@]}"
  echo "=== $tag exit $? $(date +%T)"
done
echo done > $O/DONE.$(date +%H%M%S)
