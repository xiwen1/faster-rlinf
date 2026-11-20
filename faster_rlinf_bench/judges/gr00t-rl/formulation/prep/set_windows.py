"""Formulation: window lengths set by hand (F1's probe fitted 8 iterations from a disturbed, cold latency)."""
import json
from pathlib import Path
p = Path("$WORK/gr00t-rl/ws/bench/bench.json")
c = json.loads(p.read_text())
for item in c["timing"]["plan"]:
    item["window_iters"] = {8: 25, 1: 30}[item["batch"]]
p.write_text(json.dumps(c, indent=1) + "\n")
print(c["timing"]["plan"])
