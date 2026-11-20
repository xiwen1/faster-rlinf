"""Acceptance table: every check.json under gr00t-rl/.prep/acc*/ (or prep-state/acc*/), with the expected verdict."""
import json
import sys
from pathlib import Path

EXPECTED = {
    "a_unmodified-quick": "PASS", "a_unmodified-full": "UNAVAILABLE", "a_unmodified-quick-final": "PASS",
    "b_entry_point-quick": "ERROR", "c1_skip_layers-quick": "INCORRECT", "c1_skip_layers-full": "INCORRECT",
    "c2_extra_rng-quick": "INCORRECT", "c3_timed_only-quick": "PASS", "c3_timed_only-full": "INCORRECT",
    "c4_py_random-quick": "INCORRECT", "c4b_np_random-quick": "INCORRECT", "c5_replay_only-quick": "INCORRECT",
    "c6_logprob_only-quick": "INCORRECT", "c7_perturb-quick": "INCORRECT", "d_equivalent-quick": "PASS",
    "d_equivalent-full": "UNAVAILABLE", "f_slower-quick": "PASS", "f_slower-full": "UNAVAILABLE",
}
root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("acc*/*/check.json")):
    tag = f.parent.name
    if ".prev-" in tag:
        continue
    r = json.loads(f.read_text())
    corr, perf = r.get("correctness") or {}, r.get("performance") or {}
    num = corr.get("numerics") or {}
    sp = perf.get("speedup", perf.get("measured_speedup"))
    detail = corr.get("verdict") or corr.get("stage") or ""
    if r["status"] == "ERROR":
        detail = (r.get("note") or corr.get("note") or str(corr.get("problems") or ""))[:90]
    if num.get("exact_failures"):
        detail += f" exact_failures={str(num['exact_failures'])[:80]}"
    elif num.get("checked"):
        detail += f" bitwise {num.get('bitwise_identical')}/{num.get('checked')} tier {num.get('measured_tier')}"
    to = perf.get("timed_outputs") or {}
    if to:
        detail += f" timed_equal={to.get('candidate_equals_reference')}"
    exp = EXPECTED.get(tag, "?")
    rows.append((tag, r["status"], exp, "yes" if r["status"] == exp else "NO", sp, detail, f.parent.parent.name))
print("| candidate | verdict | expected | match | speedup | detail | run |")
print("|---|---|---|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
