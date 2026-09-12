"""Why does code execution grade differently on a compute node than on the login node?"""
import collections, os, subprocess, sys
sys.path.insert(0, "src")
from bidir import domains, prompts
from bidir.schema import read_pairs
from bidir.config import load_config, DATA_DIR

print("=== environment ===")
print("node        ", os.uname().nodename)
print("nproc       ", os.cpu_count(), "| sched_affinity", len(os.sched_getaffinity(0)))
print("TMPDIR      ", os.environ.get("TMPDIR"))
for f in ("/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/memory.max"):
    try: print(f"{f:<28}", open(f).read().strip())
    except Exception as e: print(f"{f:<28} unreadable ({type(e).__name__})")
try:
    import resource
    for nm in ("RLIMIT_NPROC", "RLIMIT_AS", "RLIMIT_CPU"):
        print(f"  {nm:<14}", resource.getrlimit(getattr(resource, nm)))
except Exception as e:
    print("  rlimit:", e)
print("  can spawn a child python:", subprocess.run(
    [sys.executable, "-c", "print(1)"], capture_output=True, text=True).stdout.strip() or "NO")

for cell in ("code", "coverage", "exec"):
    mod = domains.get(cell)
    cfg = dict(load_config(f"domains/{cell}.yaml"))
    insts = [p.model_dump() for p in read_pairs(DATA_DIR / cell / "test.jsonl")[:40]]
    for workers in (32, 4, 1):
        c = {**cfg, "exec_workers": workers, "coverage_workers": workers}
        for d in ("forward", "reverse"):
            gold = [prompts.completion_for(i, d) for i in insts]
            try:
                sc = mod.score_batch(d, gold, insts, c)
            except Exception as e:
                print(f"{cell}/{d} workers={workers}: RAISED {type(e).__name__}: {e}")
                continue
            n = len(sc)
            st = collections.Counter(r.get("exec_status") or r.get("coverage_status") for r in sc)
            print(f"{cell:<9}/{d:<8} workers={workers:<3} strict={sum(r['strict'] for r in sc)/n:.3f} "
                  f"off_target={sum(r.get('off_target',0) for r in sc)/n:.3f} "
                  f"echo={sum(r.get('echo',0) for r in sc)/n:.3f} statuses={dict(st)}")
            # first failing row, verbatim, so the cause is visible rather than inferred
            bad = [r for r in sc if not r["strict"]]
            if bad and workers == 1:
                r = bad[0]
                keys = [k for k in r if k not in ("criterion",)]
                print("    first failing row:", {k: (str(r[k])[:90]) for k in keys})
