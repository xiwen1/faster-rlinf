"""Dump every scalar tensorboard metric of a run to <run>/metrics.json as {tag: {step: value}}."""
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
out = {}
for f in sorted(glob.glob(os.path.join(run, "logs", "**", "events.out.tfevents*"), recursive=True)):
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            out.setdefault(tag, {})[str(e.step)] = e.value
with open(os.path.join(run, "metrics.json"), "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(run, len(out), "tags")
