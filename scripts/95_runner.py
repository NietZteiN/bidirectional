#!/usr/bin/env python
"""Hold a fixed number of jobs in flight and feed them from the priority queue.

    python scripts/95_runner.py --max-inflight 4            # top up to 4 and exit
    python scripts/95_runner.py --max-inflight 4 --dry-run
    python scripts/95_runner.py --status                    # what is in flight and what is next

WHY A RUNNER. Measured over the 64 h to 2026-09-14 this project achieved 0.89 GPU-hours per
wall-hour -- under one sustained GPU on a cluster with ~65 -- because work was submitted in
bursts and then the queue drained while nobody was looking. The remaining ~378 GPU-hours take
~18 days at that rate. Holding a steady allocation is worth more than any speedup to the code:
at 3 sustained slots it is ~5 days.

THE DEFAULT IS 3, not 4, and deliberately. Four other projects share this account and this
cluster; three is what this project takes. Raise it with --max-inflight only if the share has
actually been renegotiated, the same discipline $BIDIR_JUNO_SHARE applies to the juno pool.

IT PREFERS THE UNCAPPED PARTITIONS. `h100` and `a30` carry no QoS, so four jobs there cost the
`juno` pool nothing and squeeze no neighbouring project -- the account shares that pool of 4
across four projects. Only cells that genuinely need a big card (two resident models, or an
8B+ model) are placed on `h200`, and never more than $BIDIR_JUNO_SHARE of them at once.

IT IS IDEMPOTENT. Every invocation recomputes what is missing from disk: a cell whose adapters
are all complete and whose eval exists is never resubmitted, and a cell already in the queue is
counted, not duplicated. Safe to re-run as often as you like; the hard cap is enforced by
counting, not by memory.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as A  # noqa: E402
from bidir.config import RESULTS_DIR, RUNS_DIR  # noqa: E402
from bidir.train import adapter_dir  # noqa: E402

SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

#: Priority order, from TASKS.md. Each entry is (label, cells, models, seeds).
#: Cells are only attempted when their base gate has PASSED -- an ungated or failing cell is
#: reported as blocked rather than silently skipped, because a gate failure is a finding.
CORE = ["mt_de-en", "mt_en-de", "sql", "code"]
READY = ["relation", "mt_en-zh", "mt_zh-en"]
FORMAT_BLOCKED = ["diacritics", "algebra", "algebra_rev", "fmt", "fmt_det75"]

PLAN = [
    ("P1 s17 breadth",        READY,          ["llama32-3b"], [17]),
    ("P2 blocked-cell s17",   FORMAT_BLOCKED, ["llama32-3b"], [17]),
    ("P3 seeds",              CORE,           ["llama32-3b"], [42, 1234]),
    ("P5 gemma3-4b",          CORE + READY,   ["gemma3-4b"],  [17]),
    ("P6 olmo2-1b",           CORE + READY,   ["olmo2-1b"],   [17]),
]

#: Long-sequence or two-model cells: an a30 placement does not finish in a sane walltime.
NO_A30 = {"code", "sql", "d2t", "coverage", "exec", "relation"}
#: Needs the 141 GB card: two models resident at once.
NEEDS_H200 = {"d2t"}

OURS = re.compile(r"^(tr|ev)_(" + "|".join(sorted(set(CORE + READY + FORMAT_BLOCKED))) + r")_")


def squeue_ours() -> list[tuple[str, str]]:
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j|%T"],
                         capture_output=True, text=True, timeout=30).stdout
    rows = []
    for line in out.splitlines():
        name, _, state = line.partition("|")
        if OURS.match(name.strip()):
            rows.append((name.strip(), state.strip()))
    return rows


def adapter_complete(domain: str, model: str, arm: str, seed: int) -> bool:
    spec = A.resolve(arm)
    out = adapter_dir(domain, model, arm, 32, seed,
                      root="adapters_fullft" if spec.full_ft else "adapters")
    mf = out / "run_manifest.json"
    if not (out / "final").exists() or not mf.exists():
        return False
    try:
        return bool((json.loads(mf.read_text()).get("adapter") or {}).get("sha256"))
    except Exception:
        return False


def gate_passed(domain: str, model: str) -> bool | None:
    files = sorted((RESULTS_DIR / "base_gates" / domain).glob(f"{model}_*.json"))
    if not files:
        return None
    try:
        return bool(json.loads(files[-1].read_text()).get("passes_gate"))
    except Exception:
        return None


def cell_state(domain: str, model: str, seed: int) -> tuple[list[str], bool]:
    """(arms still to train, whether an eval already exists)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "slurm" / "pipeline_grid.py")
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    wanted = pg.resolvable_arms(list(A.TIERS["full"]), domain, model)[0]
    todo = [a for a in wanted if not adapter_complete(domain, model, a, seed)]
    evald = bool(list(RESULTS_DIR.glob(f"*/{domain}/{model}/*_s{seed}/trials.jsonl")))
    return todo, evald


def submit(name, argv, partition, time, dep=None, cpus=16, mem="64G", dry=False):
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem, "--cpus", str(cpus)]
    if dep:
        cmd += ["--dependency", f"afterok:{dep}"]
    if partition != "a30":
        cmd += ["--exclude", "g-06-01"]
    if dry:
        cmd.append("--dry-run")
    cmd += ["--argv"] + argv
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if dry:
        return "DRY"
    for line in r.stdout.splitlines():
        if line.startswith("submitted "):
            return line.split()[1]
    sys.stderr.write(r.stdout + r.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-inflight", type=int, default=3,
                    help="cells in flight at once. A cell is one train job plus its eval, so "
                         "this is the number of GPUs held, not the number of jobs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "slurm" / "pipeline_grid.py")
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)

    inflight = squeue_ours()
    running = [n for n, s in inflight if s == "RUNNING"]
    n_cells = len({n.split("_", 1)[1] for n, _ in inflight})
    print(f"[runner] in flight: {n_cells} cell(s), {len(inflight)} job(s), "
          f"{len(running)} running")
    for n, s in sorted(inflight):
        print(f"    {s:<10} {n}")

    todo_list, blocked = [], []
    for label, cells, models, seeds in PLAN:
        for model in models:
            for seed in seeds:
                for cell in cells:
                    g = gate_passed(cell, model)
                    if g is not True:
                        blocked.append((label, cell, model,
                                        "gate not run" if g is None else "gate FAILED"))
                        continue
                    todo, evald = cell_state(cell, model, seed)
                    if not todo and evald:
                        continue
                    todo_list.append((label, cell, model, seed, todo, evald))

    print(f"[runner] {len(todo_list)} cell(s) ready, {len(blocked)} blocked")
    if a.status:
        for label, cell, model, seed, todo, evald in todo_list[:12]:
            print(f"    READY   {label:<18} {cell}/{model}/s{seed}  "
                  f"{len(todo)} arm(s) to train{'' if evald else ', eval needed'}")
        for label, cell, model, why in blocked[:12]:
            print(f"    BLOCKED {label:<18} {cell}/{model}: {why}")
        return 0

    free = max(0, a.max_inflight - n_cells)
    if not free:
        print(f"[runner] at capacity ({a.max_inflight}); nothing submitted")
        return 0

    launched = 0
    for label, cell, model, seed, todo, evald in todo_list:
        if launched >= free:
            break
        part = "h200" if cell in NEEDS_H200 else ("h100" if cell in NO_A30 else "h100,a30")
        jid = None
        if todo:
            units = A.units(todo)
            hours = min(47.0, max(2.0, units * pg._per_unit_hours(cell, "a30" if "a30" in part else "h100") * 1.6))
            t = f"{int(hours):02d}:{int((hours - int(hours)) * 60):02d}:00"
            jid = submit(f"tr_{cell}_{model}_s{seed}",
                         ["scripts/20_train_pack.py", "--domain", cell, "--model", model,
                          "--seed", str(seed), "--arms", "full"], part, t, dry=a.dry_run)
            print(f"[runner] {label}: train {cell}/{model}/s{seed} on {part} "
                  f"({len(todo)} arms, {t}) -> {jid}")
            if jid is None:
                continue                      # refused; do not orphan an eval behind it
        arms = ",".join(["base"] + pg.resolvable_arms(list(A.TIERS["full"]), cell, model)[0])
        ev = submit(f"ev_{cell}_{model}_s{seed}",
                    ["-m", "bidir.evaluate", "--domain", cell, "--model", model,
                     "--seed", str(seed), "--arms", arms, "--tag", "small"],
                    part, "03:00:00", dep=jid, dry=a.dry_run)
        print(f"[runner] {label}: eval  {cell}/{model}/s{seed} -> {ev}")
        launched += 1

    print(f"[runner] launched {launched} cell(s); {n_cells + launched}/{a.max_inflight} in flight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
