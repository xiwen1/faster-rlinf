"""Split py-spy raw samples of the actor's run_training into phases (seconds at the sampling rate)."""
import collections, sys
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
CATS = [("load_param", "load_param_and_grad"), ("load_optim", "load_optimizer"),
        ("offload", "offload_"), ("backward", "backward"), ("optim_step", "optimizer_step"),
        ("optim_step", ".step ("), ("forward", "forward"), ("get_batch", "get_batch"),
        ("recv", "recv"), ("sync", "synchronize")]
c = collections.Counter(); leaf = collections.Counter(); tot = 0
for line in open(sys.argv[1]):
    line = line.rstrip()
    if "run_training" not in line: continue
    stack, n = line.rsplit(" ", 1); n = int(n)
    if "EmbodiedFSDPActor.run_training" not in stack: continue
    tot += n
    for name, key in CATS:
        if key in stack:
            c[name] += n; break
    else:
        c["other"] += n
    leaf[stack.split(";")[-1][:110]] += n
print(f"run_training samples {tot} = {tot/RATE:.0f} s")
for k, v in c.most_common(): print(f"  {k:11s} {v/RATE:7.1f} s  {100*v/tot:5.1f}%")
print("  top leaves:")
for k, v in leaf.most_common(8): print(f"    {v/RATE:7.1f} s  {k}")
