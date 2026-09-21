#!/usr/bin/env python
"""Hold a fixed number of jobs in flight and feed them from the priority queue.

    python scripts/95_runner.py                             # top up to 12 (the default) and exit
    python scripts/95_runner.py --dry-run
    python scripts/95_runner.py --status                    # what is in flight and what is next

WHY A RUNNER. Measured over the 64 h to 2026-09-14 this project achieved 0.89 GPU-hours per
wall-hour -- under one sustained GPU on a cluster with ~65 -- because work was submitted in
bursts and then the queue drained while nobody was looking. The remaining ~378 GPU-hours take
~18 days at that rate. Holding a steady allocation is worth more than any speedup to the code:
at 3 sustained slots it is ~5 days.

THE DEFAULT IS 12, raised from 7 on 2026-09-21. The ceiling is not one number, because the
limits are asymmetric. Queried from Slurm rather than assumed:

    juno QoS (h200 + normal)   MaxJobsPU=4      hard, and PER USER -- so it is shared across
                               MaxSubmitJobsPU=100   every project this account runs
    h100, a30                  no job-count limit    bounded only by physical contention

So the real cap is 4 on h200 plus whatever the uncapped partitions yield: ~9 h100 GPUs and 4
a30s plus MIG slices, contested by ~50 users. 12 is 4 + 8 opportunistic. Queueing beyond what
we can win costs nothing (MaxSubmitJobsPU is 100) but gains nothing either.

$BIDIR_JUNO_SHARE stays at 3 of the 4, NOT 4. The pool is per user, so taking all of it would
starve this account's other projects on h200 -- and `probing` has an ARR deadline of 2026-10-12,
which is nearer than ours. That is a scheduling courtesy, not a technical requirement; raise it
if the priorities change.

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

#: Cells where `sft` actually collapsed the reverse direction, so "erased or merely suppressed?"
#: is a question with something to measure. Taken from the recorded verdicts, not from intuition:
#: each of these meets the pre-registered criterion (<= -50 % relative on strict reverse).
COLLAPSED = ["code", "sql", "fmt", "mt_de-en", "mt_zh-en", "mt_en-zh"]
#: Same criterion, applied to each model's own recorded verdict -- collapse is NOT a property of
#: the cell alone. `sql` collapses on llama32-3b (-53.7 %) and not on gemma3-4b (-4.8 %), and
#: `relation` never collapses anywhere. Running a relearning ladder where nothing collapsed
#: would measure recovery from a loss that did not happen.
GEMMA_COLLAPSED = ["code", "mt_de-en", "mt_en-de", "mt_en-zh", "mt_zh-en"]
OLMO_COLLAPSED = ["mt_de-en", "mt_en-de"]

#: (label, cells, models, seeds, tier, tag). `tier` picks the arm set; `tag` names the run
#: directory, and MUST differ between tiers -- a mechanism eval scores a different arm set, and
#: writing it to the main tag would replace a 12-arm result with a 6-arm one.
PLAN = [
    ("P1 s17 breadth",        READY,          ["llama32-3b"], [17],      "full",    "small"),
    ("P2 blocked-cell s17",   FORMAT_BLOCKED, ["llama32-3b"], [17],      "full",    "small"),
    ("P3 seeds",              CORE,           ["llama32-3b"], [42, 1234], "full",   "small"),
    # Tag is "relearn", NOT "mech": `53_mech_report.py` globs `*/*/*/relearn_s*` and would not
    # have seen a single one of these runs. The consumer's convention predates this entry.
    ("P4 mechanism",          COLLAPSED,      ["llama32-3b"], [17],      "relearn", "relearn"),
    # The control the P4 curves are READ AGAINST. Without it "fast relearning" has no scale and
    # `53_mech_report` cannot return a verdict -- it defaulted to "erased" when this was absent.
    # Needs `sft` first (forward on the invented transform) before the ladder has a start point.
    ("P4 never-had control",  ["fmt_novel"],  ["llama32-3b"], [17],      "core",    "small"),
    ("P4 control ladder",     ["fmt_novel"],  ["llama32-3b"], [17],      "relearn", "relearn"),
    ("P5 gemma3-4b",          CORE + READY,   ["gemma3-4b"],  [17],      "full",    "small"),
    ("P6 olmo2-1b",           CORE + READY,   ["olmo2-1b"],   [17],      "full",    "small"),

    # --- added 2026-09-21 -----------------------------------------------------------------
    # Does "suppressed, not erased" replicate off llama32-3b? The control is MODEL-SPECIFIC --
    # tau and the never-had curve are both read from the model's own base -- so each model needs
    # its own fmt_novel before its ladder means anything. Collapsed cells are taken from the
    # recorded verdicts: gemma3-4b collapses on 5, olmo2-1b on 2.
    ("P4b control gemma",     ["fmt_novel"],  ["gemma3-4b"],  [17],      "core",    "small"),
    ("P4b ladder gemma",      ["fmt_novel"],  ["gemma3-4b"],  [17],      "relearn", "relearn"),
    ("P4b mechanism gemma",   GEMMA_COLLAPSED, ["gemma3-4b"], [17],      "relearn", "relearn"),
    ("P4c control olmo",      ["fmt_novel"],  ["olmo2-1b"],   [17],      "core",    "small"),
    ("P4c ladder olmo",       ["fmt_novel"],  ["olmo2-1b"],   [17],      "relearn", "relearn"),
    ("P4c mechanism olmo",    OLMO_COLLAPSED, ["olmo2-1b"],   [17],      "relearn", "relearn"),

    # Seed replication on the second and third models.
    ("P5b gemma3-4b s42",     CORE + READY,   ["gemma3-4b"],  [42],      "full",    "small"),
    ("P6b olmo2-1b s42",      CORE + READY,   ["olmo2-1b"],   [42],      "full",    "small"),

    # P7, the scale question. Gates are queued; cells stay blocked until they pass.
    ("P7 llama31-8b",         CORE + READY,   ["llama31-8b"], [17],      "full",    "small"),
    ("P7 gemma3-12b",         CORE + READY,   ["gemma3-12b"], [17],      "full",    "small"),
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


#: Cells whose base gate PRECONDITION IS INVERTED. `fmt_novel` encodes the same documents as
#: `fmt` in an invented format no pretraining corpus contains, so the base scores at floor by
#: construction -- that is the point, not a defect. It is the never-had baseline the relearning
#: curves are read against, and a cell the base can already do would be useless as one. So the
#: normal clause ("the base can do this") is replaced by its mirror ("the base cannot"), checked
#: rather than assumed: the gate still runs and its rates must be at floor in BOTH directions.
CONTROL_CELLS = {"fmt_novel"}
CONTROL_FLOOR = 0.10


def control_gate_ok(domain: str, model: str) -> bool | None:
    """For a never-had control: the gate must have RUN and found the base at floor."""
    files = [f for f in sorted((RESULTS_DIR / "base_gates" / domain).glob(f"{model}_*.json"))
             if "_failures_" not in f.name]
    for f in reversed(files):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        dirs = d.get("directions") or {}
        if not dirs:
            continue
        rates = [v.get("rate", v.get("rate_uncapped", 1.0)) for v in dirs.values()]
        return bool(rates) and max(rates) < CONTROL_FLOOR
    return None


def gate_passed(domain: str, model: str) -> bool | None:
    # Defence in depth: dumps now live in a `failures/` subdirectory, but this glob is what
    # decides whether a cell is allowed to run, so it also refuses anything without a verdict
    # rather than trusting the newest filename.
    files = [f for f in sorted((RESULTS_DIR / "base_gates" / domain).glob(f"{model}_*.json"))
             if "_failures_" not in f.name]
    if not files:
        return None
    for f in reversed(files):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "passes_gate" in d:
            return bool(d["passes_gate"])
    return None


def cell_state(domain: str, model: str, seed: int, tier: str = "full",
               tag: str = "small") -> tuple[list[str], bool]:
    """(arms still to train, whether an eval already exists).

    TAG-SCOPED. The mechanism tier writes its own run directory: its eval scores base, sft and
    the relearn ladder, and pointing it at the main run's tag would overwrite the 12-arm result
    with a 6-arm one -- contrasts are only valid within a pass, so the two cannot be merged and
    the survivor would be whichever ran last.
    """
    wanted = resolvable(domain, model, tier)
    todo = [a for a in wanted if not adapter_complete(domain, model, a, seed)]
    evald = bool(list(RESULTS_DIR.glob(f"*/{domain}/{model}/{tag}_s{seed}/trials.jsonl")))
    return todo, evald


def job_key(cell: str, model: str, seed: int, tag: str = "small") -> str:
    """The identity a job name encodes, built in ONE place.

    `squeue_ours` recovers this by stripping the `tr_`/`ev_` prefix, so it has to be assembled
    exactly as the name is or the in-flight check compares two different strings and always
    misses. Keep this and the `submit(...)` name arguments in step.
    """
    return f"{cell}_{model}_s{seed}" + ("" if tag == "small" else f"_{tag}")


def measured_floor_hours(cell: str, model: str, tag: str = "small") -> float:
    """Never ask for less walltime than a previous run of this cell actually used.

    The analytic estimate is a table of constants (`_BASE_UNIT_H`, `_A30_FACTOR`, a long-sequence
    multiplier) and it has been wrong in the dangerous direction: `fmt_novel`'s core pack was
    sized at 4:48 on a30 and the walltime killed it with one arm left. Constants can be re-tuned
    for ever; what a job of this exact shape DID take is not a guess.

    Used as a FLOOR, never a cap, and deliberately crude: a run that was itself resumed did less
    work than a fresh one, so this can only be an underestimate of the true need -- which is
    still strictly better than a number that has already proved too small. Returns 0.0 when
    there is nothing to learn from.
    """
    try:
        out = subprocess.run(
            ["sacct", "-u", getpass.getuser(), "-S", "2026-09-01", "-X", "-P",
             "-o", "JobName,State,Elapsed"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 0.0
    # TIER-SCOPED. Job names carry the tag, and a relearn pack is a different animal from a
    # full-tier pack of the same cell: 4 cheap arms against 11. Matching on the cell alone made
    # a 4.4 h full-tier run the floor for a 0.22 h relearn pack -- a 20x over-request. The
    # suffix is the only thing that distinguishes them.
    suffix = "" if tag == "small" else f"_{tag}"
    best = 0.0
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        name = parts[0]
        m = re.fullmatch(rf"tr_{re.escape(cell)}_{re.escape(model)}_s\d+{re.escape(suffix)}", name)
        if not m:
            continue
        if parts[1] not in ("COMPLETED", "TIMEOUT"):
            continue
        el = parts[2].split("-")[-1].split(":")
        try:
            h = int(el[0]) + int(el[1]) / 60 + int(el[2]) / 3600
        except (ValueError, IndexError):
            continue
        if "-" in parts[2]:                      # D-HH:MM:SS
            h += int(parts[2].split("-")[0]) * 24
        best = max(best, h)
    return best


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


def resolvable(domain: str, model: str, tier: str = "full") -> list[str]:
    """The arms this cell can actually express -- the ONE definition both call sites use.

    `resolvable_arms` drops dose rungs whose reversed-pair count falls below one effective
    batch. The runner used to call it for the eval's `--arms` and for the walltime, but hand
    the trainer the literal string "full": `relation` (504 train pairs) then had mix1/mix5/mix10
    trained and never scored, on a walltime sized for the eight arms that survive. Both call
    sites now read this, so the set that is SIZED, the set that is TRAINED and the set that is
    SCORED cannot drift apart again.
    """
    return _pg().resolvable_arms(list(A.TIERS[tier]), domain, model)[0]


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
    ap.add_argument("--max-inflight", type=int, default=12,
                    help="cells in flight at once (default 12). A cell is one train job plus its "
                         "eval, so this is the number of GPUs held, not the number of jobs. "
                         "12 = the juno QoS hard cap of 4 on h200, plus ~8 opportunistic on the "
                         "uncapped h100/a30 partitions, which carry no job-count limit.")
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
            todo_list.append(("force " + a.force_arm, cell, model, seed, [a.force_arm], False,
                              "full", "small"))
        print(f"[runner] {len(todo_list)} cell(s) hold a stale {a.force_arm!r} adapter")
        if a.status:
            for label, cell, model, seed, *_ in todo_list:
                print(f"    FORCE   {cell}/{model}/s{seed}")
            return 0
    for label, cells, models, seeds, tier, tag in PLAN:
        if a.force_arm:
            break                     # force mode built todo_list from disk above
        for model in models:
            for seed in seeds:
                for cell in cells:
                    if cell in CONTROL_CELLS:
                        g = control_gate_ok(cell, model)
                        why = ("gate not run" if g is None else
                               f"base is NOT at floor (>={CONTROL_FLOOR}); useless as a never-had control")
                    else:
                        g = gate_passed(cell, model)
                        why = "gate not run" if g is None else "gate FAILED"
                    if g is not True:
                        blocked.append((label, cell, model, why))
                        continue
                    # THE KEY MUST BE BUILT THE SAME WAY THE JOB NAME IS. Adding the `_mech`
                    # tag suffix to job names broke this comparison silently: the guard asked for
                    # `code_llama32-3b_s17` while the queue held `code_llama32-3b_s17_mech`, so
                    # every mechanism cell looked absent and `code` was submitted twice.
                    if job_key(cell, model, seed, tag) in inflight_cells:
                        continue          # already queued or running; never submit it twice
                    todo, evald = cell_state(cell, model, seed, tier, tag)
                    if not todo and evald:
                        continue
                    todo_list.append((label, cell, model, seed, todo, evald, tier, tag))

    print(f"[runner] {len(todo_list)} cell(s) ready, {len(blocked)} blocked")
    if a.status:
        for label, cell, model, seed, todo, evald, tier, tag in todo_list[:12]:
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
    for label, cell, model, seed, todo, evald, tier, tag in todo_list:
        if launched >= free:
            break
        part = partition_for(cell)
        jid = None
        if todo:
            units = A.units(todo)
            # SAFETY MARGIN, MEASURED NOT GUESSED. This was a flat 1.6 and it was not enough on
            # a30: `fmt_novel`'s 5-arm core pack was sized at 4:48 and completed 4 of 5 arms
            # before the walltime killed it (2026-09-20). Measured medians are 0.80 h/job on
            # h100 and 2.22 h on a30 -- a ratio of 2.8, so _A30_FACTOR=3.0 is sound and the
            # shortfall was the margin, not the rate. A timeout costs a whole queue wait (median
            # 40-160 min) and the resume only preserves finished arms, whereas an over-long
            # request costs nothing while we are waiting on Priority rather than backfill. So
            # the margin is wider where the variance is: a30's p90/median spread is 4.4x.
            on_a30 = "a30" in part
            margin = 2.2 if on_a30 else 1.8
            est = units * pg._per_unit_hours(cell, "a30" if on_a30 else "h100") * margin
            # The measured floor is for whatever pack that run held; scale it by the share of
            # this tier still to do, or a one-arm resume would request the whole pack's hours.
            full_units = A.units(resolvable(cell, model, tier)) or units
            floor = measured_floor_hours(cell, model, tag) * 1.3 * min(1.0, units / full_units)
            if floor > est:
                print(f"[runner] {cell}/{model}: estimate {est:.2f} h below measured floor "
                      f"{floor:.2f} h (a previous run of this cell took longer); using the floor")
            hours = min(47.0, max(2.0, est, floor))
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
            jid = submit("tr_" + job_key(cell, model, seed, tag), pack, part, t, dry=a.dry_run)
            print(f"[runner] {label}: train {cell}/{model}/s{seed} on {part} "
                  f"({len(todo)} arms, {t}) -> {jid}")
            if jid is None:
                continue                      # refused; do not orphan an eval behind it
        # `sft` is carried into a mechanism eval even though it is not in the relearn tier:
        # the ladder is read against the collapse it starts from, and a contrast is only valid
        # within one pass, so it cannot be borrowed from the main run's trials.
        extra = ["sft"] if tier == "relearn" else []
        arms = ",".join(["base"] + extra + resolvable(cell, model, tier))
        ev_time = "06:00:00" if cell in SLOW_EVAL else "03:00:00"
        ev = submit("ev_" + job_key(cell, model, seed, tag),
                    ["-m", "bidir.evaluate", "--domain", cell, "--model", model,
                     "--seed", str(seed), "--arms", arms, "--tag", tag],
                    part, ev_time, dep=jid, dry=a.dry_run)
        print(f"[runner] {label}: eval  {cell}/{model}/s{seed} -> {ev}")
        launched += 1

    print(f"[runner] launched {launched} cell(s); {n_cells + launched}/{a.max_inflight} in flight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
