#!/usr/bin/env python
"""Hold a fixed number of jobs in flight and feed them from the priority queue.

    python scripts/95_runner.py                             # top up to 3 (the default) and exit
    python scripts/95_runner.py --dry-run
    python scripts/95_runner.py --status                    # what is in flight and what is next

WHY A RUNNER. Measured over the 64 h to 2026-09-14 this project achieved 0.89 GPU-hours per
wall-hour -- under one sustained GPU on a cluster with ~65 -- because work was submitted in
bursts and then the queue drained while nobody was looking. The remaining ~378 GPU-hours take
~18 days at that rate. Holding a steady allocation is worth more than any speedup to the code:
at 3 sustained slots it is ~5 days.

THE DEFAULT IS 7, raised from 3 on 2026-09-18 when obtune finished and released its share of
the account. It is a NEGOTIATED number, not a technical limit: the account is shared, and the
figure should drop again if another project starts up. $BIDIR_JUNO_SHARE applies the same
discipline to the h200/normal QoS pool, which is a separate and finer-grained budget.

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
import functools
import getpass
import json
import re
import subprocess
import sys
from datetime import datetime
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
#: MT is here for the EVAL's sake, not the training's: its criterion scores COMET-22 on the same
#: card, and vLLM at 0.85 of an a30's 24 GB leaves ~3.6 GB -- too little, so COMET falls back to
#: CPU and a 24,288-generation pass runs past any sane walltime. Both s42 MT evals timed out at
#: 3 h that way on 2026-09-15. On an h100 the same 15 % is 12 GB and COMET stays on the GPU.
NO_A30 = {"code", "sql", "d2t", "coverage", "exec", "relation",
          "mt_en-de", "mt_de-en", "mt_en-zh", "mt_zh-en"}

#: Cells whose eval scores a second neural metric (COMET) and therefore needs room for it.
#: 6 h rather than 3: 12 arms x 1,012 instances x 2 directions is 24k generations plus scoring.
SLOW_EVAL = {"mt_en-de", "mt_de-en", "mt_en-zh", "mt_zh-en", "d2t"}
#: Needs the 141 GB card: two models resident at once.
NEEDS_H200 = {"d2t"}

OURS = re.compile(r"^(tr|ev)_(" + "|".join(sorted(set(CORE + READY + FORMAT_BLOCKED))) + r")_")


#: Reasons a PENDING job will never start on its own. A job in one of these states occupies a
#: slot in the in-flight count for ever, and the count is what gates submission -- on 2026-09-17
#: the `fmt` cell was a user-held train job, a DependencyNeverSatisfied orphan and an eval
#: waiting on the held one, so it held 1 of 3 slots with nothing that could run, and the runner
#: reported "at capacity (3)" while exactly one job was running. Capacity must mean work.
DEAD_REASONS = ("DependencyNeverSatisfied", "JobHeldUser", "JobHeldAdmin")


def squeue_ours(include_dead: bool = False) -> list[tuple[str, str]]:
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%r|%E"],
                         capture_output=True, text=True, timeout=30).stdout
    jobs = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 5:
            continue
        jid, name, state, reason, dep = parts[0], parts[1], parts[2], parts[3], parts[4]
        if OURS.match(name):
            jobs.append({"id": jid, "name": name, "state": state, "reason": reason, "dep": dep})

    # DEADNESS IS TRANSITIVE. A job waiting on a job that can never run can never run either,
    # and Slurm does not say so: its reason stays plain "Dependency". `ev_fmt` was queued behind
    # a user-HELD `tr_fmt`, so it read as an ordinary pending job and kept the `fmt` cell in the
    # in-flight count -- 3 cells "at capacity" with exactly 1 job running. Closes over the
    # dependency graph so the whole stalled chain drops out of the count, not just its head.
    dead = {j["id"] for j in jobs if j["reason"] in DEAD_REASONS}
    for _ in range(len(jobs)):
        grew = False
        for j in jobs:
            if j["id"] in dead:
                continue
            if any(d in dead for d in re.findall(r"\d+", j["dep"])):
                dead.add(j["id"])
                grew = True
        if not grew:
            break
    return [(j["name"], j["state"]) for j in jobs if include_dead or j["id"] not in dead]


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
    wanted = resolvable(domain, model)
    todo = [a for a in wanted if not adapter_complete(domain, model, a, seed)]
    evald = bool(list(RESULTS_DIR.glob(f"*/{domain}/{model}/*_s{seed}/trials.jsonl")))
    return todo, evald


def partition_for(cell: str) -> str:
    """Which partitions this cell may be placed on, most-permissive first.

    h200 IS WHERE THE FREE CAPACITY IS. The cluster's 52 h200s sit mostly idle behind the `juno`
    QoS (4 concurrent for the WHOLE account); the uncapped partitions are ~9 GPUs contended by
    every other user. A cell that cannot use an a30 was pinned to "h100" alone and so could not
    touch the capacity that actually exists. It now asks for either and lets Slurm take whichever
    frees first -- `submit.py` still enforces $BIDIR_JUNO_SHARE, and when that budget is full it
    drops the h200 option rather than refusing, so this never costs a cell an h100 it could have
    had.
    """
    if cell in NEEDS_H200:
        return "h200"
    if cell in NO_A30:
        return "h100,h200"
    return "h100,a30"


def find_adapters(arm: str, stale_before: float | None = None) -> list[tuple[str, str, int]]:
    """Every (cell, model, seed) holding a COMPLETE adapter for `arm`, read off disk.

    Used by --force-arm, where the question is "which cells hold the withdrawn thing", not
    "what is still scheduled". Matches the layout `adapter_dir` writes:
    `adapters/<cell>/<model>/<arm>_r<rank>_s<seed>/final`.

    `stale_before` IS WHAT MAKES THE SWEEP TERMINATE. Selecting "cells that have the adapter"
    re-selects a cell the moment it is rebuilt, because rebuilding it leaves it having the
    adapter. Run in a loop with one free slot that is not a sweep, it is a treadmill: overnight
    on 2026-09-17 it retrained `algebra/s17` (first in sort order), watched it complete, and
    retrained it again -- ten other stale cells untouched, for nine hours. The cutoff is the
    only thing that distinguishes "holds the arm" from "holds the OLD arm".
    """
    out: list[tuple[str, str, int]] = []
    root = RUNS_DIR / "adapters"
    if not root.is_dir():
        return out
    for d in root.glob(f"*/*/{arm}_r*_s*"):
        if not (d / "final").is_dir():
            continue                  # half-written: the pack retrains it anyway
        if stale_before is not None and (d / "final").stat().st_mtime >= stale_before:
            continue                  # already redone since the cutoff
        m = re.fullmatch(rf"{re.escape(arm)}_r\d+_s(\d+)", d.name)
        if m:
            out.append((d.parent.parent.name, d.parent.name, int(m.group(1))))
    return sorted(set(out))


def resolvable(domain: str, model: str) -> list[str]:
    """The arms this cell can actually express -- the ONE definition both call sites use.

    `resolvable_arms` drops dose rungs whose reversed-pair count falls below one effective
    batch. The runner used to call it for the eval's `--arms` and for the walltime, but hand
    the trainer the literal string "full": `relation` (504 train pairs) then had mix1/mix5/mix10
    trained and never scored, on a walltime sized for the eight arms that survive. Both call
    sites now read this, so the set that is SIZED, the set that is TRAINED and the set that is
    SCORED cannot drift apart again.
    """
    return _pg().resolvable_arms(list(A.TIERS["full"]), domain, model)[0]


@functools.lru_cache(maxsize=1)
def _pg():
    """`scripts/slurm/pipeline_grid.py` is a script, not a package module; load it once."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "slurm" / "pipeline_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    # SURFACE THE SUBMITTER'S STDERR EVEN ON SUCCESS. This only echoed it on FAILURE, so a
    # submission that succeeded after being quietly altered said nothing at all: the note
    # explaining that h200 had been dropped from a job's partition list was captured here and
    # thrown away, and the consequence (jobs pinned to one partition overnight) was invisible
    # until someone read a .sbatch file.
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    for line in r.stdout.splitlines():
        if line.startswith("submitted "):
            return line.split()[1]
    sys.stderr.write(r.stdout)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-inflight", type=int, default=7,
                    help="cells in flight at once (default 7). A cell is one train job plus its "
                         "eval, so this is the number of GPUs held, not the number of jobs. "
                         "Seven is this project's share since obtune finished on 2026-09-18; it "
                         "is negotiated, not technical, and should drop if another project starts.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stale-before", default=None, metavar="ISO8601",
                    help="with --force-arm: treat only adapters written BEFORE this instant as "
                         "stale. Required, because 'has the adapter' re-selects a cell as soon "
                         "as it is rebuilt and the sweep never terminates.")
    ap.add_argument("--force-arm", default=None, metavar="ARM",
                    help="retrain ARM over its existing adapter in every cell that has one, then "
                         "re-evaluate. For an arm whose results have been WITHDRAWN -- the pack "
                         "skips a complete adapter forever, so this is the only way to redo one "
                         "without deleting it. See PREREGISTRATION Amendment 30 (mixedtask).")
    a = ap.parse_args()

    pg = _pg()

    inflight = squeue_ours()
    running = [n for n, s in inflight if s == "RUNNING"]
    # The KEYS, not just the count. These were only counted, and the ready list was then built
    # without consulting them, so a cell already in flight that still had an arm to train was
    # re-submitted whenever a slot looked free -- two `tr_fmt_llama32-3b_s17` jobs racing on one
    # adapter directory on 2026-09-16. Latent until now because the runner is usually called at
    # capacity and returns before this point.
    inflight_cells = {n.split("_", 1)[1] for n, _ in inflight}
    n_cells = len(inflight_cells)
    print(f"[runner] in flight: {n_cells} cell(s), {len(inflight)} job(s), "
          f"{len(running)} running")
    for n, s in sorted(inflight):
        print(f"    {s:<10} {n}")

    todo_list, blocked = [], []
    cutoff = None
    if a.force_arm:
        if not a.stale_before:
            print("[runner] --force-arm requires --stale-before ISO8601; refusing to run an "
                  "unterminating sweep", file=sys.stderr)
            return 2
        cutoff = datetime.fromisoformat(a.stale_before).timestamp()
    if a.force_arm:
        # ENUMERATED FROM DISK, NOT FROM `PLAN`. A withdrawal applies to every cell that holds
        # the arm, and `PLAN` covers only what is still scheduled -- the four original gate cells
        # (code, sql, mt_de-en, mt_en-de at s17) appear in no entry, so a PLAN-driven sweep would
        # have quietly left four stale adapters in place and called the job done.
        for d in sorted(find_adapters(a.force_arm, stale_before=cutoff)):
            cell, model, seed = d
            if f"{cell}_{model}_s{seed}" in inflight_cells:
                continue
            todo_list.append(("force " + a.force_arm, cell, model, seed, [a.force_arm], False))
        print(f"[runner] {len(todo_list)} cell(s) hold a stale {a.force_arm!r} adapter")
        if a.status:
            for label, cell, model, seed, _, _ in todo_list:
                print(f"    FORCE   {cell}/{model}/s{seed}")
            return 0
    for label, cells, models, seeds in PLAN:
        if a.force_arm:
            break                     # force mode built todo_list from disk above
        for model in models:
            for seed in seeds:
                for cell in cells:
                    g = gate_passed(cell, model)
                    if g is not True:
                        blocked.append((label, cell, model,
                                        "gate not run" if g is None else "gate FAILED"))
                        continue
                    if f"{cell}_{model}_s{seed}" in inflight_cells:
                        continue          # already queued or running; never submit it twice
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
        part = partition_for(cell)
        jid = None
        if todo:
            units = A.units(todo)
            hours = min(47.0, max(2.0, units * pg._per_unit_hours(cell, "a30" if "a30" in part else "h100") * 1.6))
            t = f"{int(hours):02d}:{int((hours - int(hours)) * 60):02d}:00"
            # The walltime above is sized from `todo`, so the pack must be asked for `todo` and
            # not for the whole `full` tier. It was asked for "full": `relation` has 504 train
            # pairs, so `resolvable_arms` drops mix1/mix5/mix10 (5/25/50 reversed pairs, all
            # under the effective batch of 64) -- the runner then sized 8 arms and trained 11.
            # Wasted GPU on three arms the eval can never score, and a walltime short by three
            # arms on every small-corpus cell. Same failure as always: what we SIZED for was not
            # what we ASKED for.
            pack = ["scripts/20_train_pack.py", "--domain", cell, "--model", model,
                    "--seed", str(seed), "--arms", ",".join(todo)]
            if a.force_arm:
                pack.append("--force")        # retrain over the stale adapter; nothing is deleted
            jid = submit(f"tr_{cell}_{model}_s{seed}", pack, part, t, dry=a.dry_run)
            print(f"[runner] {label}: train {cell}/{model}/s{seed} on {part} "
                  f"({len(todo)} arms, {t}) -> {jid}")
            if jid is None:
                continue                      # refused; do not orphan an eval behind it
        arms = ",".join(["base"] + resolvable(cell, model))
        ev_time = "06:00:00" if cell in SLOW_EVAL else "03:00:00"
        ev = submit(f"ev_{cell}_{model}_s{seed}",
                    ["-m", "bidir.evaluate", "--domain", cell, "--model", model,
                     "--seed", str(seed), "--arms", arms, "--tag", "small"],
                    part, ev_time, dep=jid, dry=a.dry_run)
        print(f"[runner] {label}: eval  {cell}/{model}/s{seed} -> {ev}")
        launched += 1

    print(f"[runner] launched {launched} cell(s); {n_cells + launched}/{a.max_inflight} in flight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
