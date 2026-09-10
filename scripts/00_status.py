#!/usr/bin/env python
"""What exists, what is running, and what is left. One screen.

    python scripts/00_status.py
    python scripts/00_status.py --tier small     # what that tier still needs

A campaign this size fails by losing track, not by any single job failing. This reads the
filesystem and squeue rather than a checklist, so it cannot drift from what is actually there.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as arm_registry  # noqa: E402
from bidir import domains  # noqa: E402
from bidir.config import DATA_DIR, RESULTS_DIR, RUNS_DIR, load_models  # noqa: E402


def squeue_rows() -> list[dict]:
    try:
        out = subprocess.run(["squeue", "-u", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip(),
                              "-h", "-o", "%i|%P|%j|%T|%r|%M"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) >= 6:
            rows.append({"id": p[0], "partition": p[1], "name": p[2], "state": p[3],
                         "reason": p[4], "time": p[5]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default=None)
    a = ap.parse_args()

    print("=== data ===")
    built = sorted(p.parent.name for p in DATA_DIR.glob("*/build_report.json"))
    total = 0
    for cell in built:
        r = json.loads((DATA_DIR / cell / "build_report.json").read_text())
        c = r["counts"]
        n = sum(c.values())
        total += n
        leak = max((v["content"] for k, v in r["overlaps"].items() if k.endswith("_x_test")), default=0)
        flag = "" if leak == 0 else f"  LEAK={leak}"
        print(f"  {cell:<14s} {c['train']:>6}/{c['val']:>4}/{c['test']:>4}{flag}")
    missing = [c for c in domains.CELLS if c not in built]
    print(f"  {len(built)} cells, {total:,} pairs" + (f"; NOT BUILT: {missing}" if missing else ""))

    print("\n=== adapters ===")
    by_cell: dict[tuple, set] = defaultdict(set)
    for p in RUNS_DIR.glob("adapters*/*/*/*/final"):
        arm_seed = p.parent.name
        by_cell[(p.parent.parent.parent.name, p.parent.parent.name)].add(arm_seed)
    if not by_cell:
        print("  none yet")
    for (domain, model), arms_ in sorted(by_cell.items()):
        print(f"  {domain:<14s} {model:<14s} {len(arms_):>3} adapters")

    print("\n=== results ===")
    runs = sorted(p.parent for p in RESULTS_DIR.glob("*/*/*/*/trials.jsonl"))
    if not runs:
        print("  none yet")
    for r in runs:
        try:
            s = json.loads((r / "summary.json").read_text())
            print(f"  {r.parent.parent.name:<14s} {r.parent.name:<14s} {r.name:<16s} "
                  f"{s.get('n_instances', '?')} inst, arms={len(s.get('arms', []))}")
        except Exception:
            print(f"  {r}  (no summary)")
    gate = RESULTS_DIR / "gate_verdict.json"
    if gate.exists():
        v = json.loads(gate.read_text())
        print(f"\n  GATE: {'PASS' if v.get('passes') else 'FAIL'} — "
              f"collapsing NLP cells: {v.get('collapsing_nlp_cells')}")

    print("\n=== queue ===")
    rows = squeue_rows()
    if not rows:
        print("  nothing queued")
    running = [r for r in rows if r["state"] == "RUNNING"]
    pending = [r for r in rows if r["state"] != "RUNNING"]
    for r in running:
        print(f"  RUNNING  {r['id']:<10s} {r['partition']:<6s} {r['name'][:34]:<34s} {r['time']}")
    caps = defaultdict(int)
    for r in pending:
        caps[r["reason"]] += 1
    for reason, n in sorted(caps.items(), key=lambda kv: -kv[1]):
        print(f"  pending  {n:>3} x {reason}")
    print(f"  {len(running)} running / {len(pending)} pending"
          + ("   [at the per-user cap]" if any("QOSMaxJobsPerUser" in r["reason"] for r in pending) else ""))

    if a.tier:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "slurm" / "pipeline_grid.py")
        pg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pg)
        t = pg.TIERS[a.tier]
        names = list(arm_registry.TIERS.get(t["arms"], ())) or t["arms"].split(",")
        names = [n for n in names if arm_registry.resolve(n).trains]
        todo = 0
        for m in t["models"]:
            for d in t["domains"]:
                for s in t["seeds"]:
                    have = by_cell.get((d, m), set())
                    todo += sum(1 for n in names if f"{n}_r32_s{s}" not in have)
        print(f"\n=== tier {a.tier} ===")
        print(f"  {len(t['models']) * len(t['domains']) * len(t['seeds'])} cells x {len(names)} arms; "
              f"{todo} adapters still to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
