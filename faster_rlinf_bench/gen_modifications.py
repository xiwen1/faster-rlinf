"""Regenerate MODIFICATIONS.md (Apache-2.0 4(b) change list) for faster-rlinf. Run from the repo root."""
import subprocess
import sys

BASE = "2348ece3df113874671478be1360ae3115b346e3"
SKIP = {"MODIFICATIONS.md"}
names = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed"}
out = subprocess.check_output(["git", "diff", "--name-status", BASE, "HEAD"], text=True)
lines = [
    "# Modifications to upstream RLinf",
    "",
    f"This fork is based on RLinf commit `{BASE}` (Apache-2.0).",
    "As required by section 4(b) of the Apache License 2.0, the files below were changed or added by faster-rlinf.",
    f"`git diff {BASE[:7]} HEAD -- <file>` shows each change; `git log {BASE[:7]}..HEAD` explains why.",
    "",
    "| Status | File |",
    "|---|---|",
]
for line in out.splitlines():
    if not line.strip():
        continue
    status, path = line.split("\t", 1)
    if path in SKIP:
        continue
    lines.append(f"| {names.get(status[0], status)} | `{path}` |")
open("MODIFICATIONS.md", "w").write("\n".join(lines) + "\n")

if len(sys.argv) > 1 and sys.argv[1] == "--readme":
    r = open("README.md").read()
    if not r.startswith("> **This is faster-rlinf**"):
        hdr = (
            "> **This is faster-rlinf**, a performance fork of RLinf that makes embodied RL training "
            "(π0.5, OpenVLA-OFT, GR00T N1.5) faster while keeping results bit-identical.\n"
            "> Read [FASTER_RLINF.md](FASTER_RLINF.md) for what is changed, measured speed-ups and how they "
            "are verified; [MODIFICATIONS.md](MODIFICATIONS.md) lists every changed file.\n"
            "> Everything below this box is the unchanged upstream RLinf README.\n\n"
        )
        open("README.md", "w").write(hdr + r)
