#!/usr/bin/env python3
"""Submit a whole tier of the grid as packed, dependency-linked jobs.

    python scripts/slurm/pipeline_grid.py --tier small --dry-run
    python scripts/slurm/pipeline_grid.py --tier large --max-submit 8

One job per (domain, model, seed) cell trains every arm in the tier back to back; one
dependent job evaluates the cell in a single vLLM pass. That is the packing rule from
RUN_PLAN.md §3.2, and it is what makes a 4-jobs-per-user cap survivable.

Placement follows GPU memory, not preference:
  * <= 4B  -> a30 (24 GB) and h100's MIG slices are fine, and both have their own per-user cap
  * 8-12B  -> h200, excluding nothing; h100's g-04-02 is the only full-fat node there
`--max-submit` exists because submitting 120 jobs at once is antisocial on a shared cluster
and buys nothing: the cap means only a handful run regardless, and a huge pending list makes
everyone else's backfill worse.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SUBMIT = ROOT / "scripts" / "slurm" / "submit.py"

from bidir import arms as arm_registry  # noqa: E402
from bidir.config import SEEDS_LARGE, SEEDS_SMALL, resolve_model  # noqa: E402

TIERS = {
    "small": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["code", "mt_en-de", "mt_de-en", "sql", "d2t", "fmt",
                    # Added 2026-09-10 (docs/CANDIDATE_DOMAINS.md): a formal domain, a domain
                    # whose forward direction is trivial, and a determined-but-hard inverse.
                    "algebra", "diacritics", "automata"],
        "seeds": list(SEEDS_SMALL), "arms": "full", "train_time": "10:00:00",
    },
    "small_zh": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["mt_en-zh", "mt_zh-en"],
        "seeds": list(SEEDS_SMALL), "arms": "core", "train_time": "06:00:00",
    },
    "ladder": {   # RQ3: the synthetic invertibility ladder
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        # `fmt_det*` varies whether the inverse is DETERMINED; `automata` is determined but
        # computationally hard. Running them together is what separates the two kinds of
        # hardness RQ3 otherwise conflates.
        "domains": ["fmt_det75", "fmt_det50", "fmt_det25", "fmt_det00", "automata"],
        "seeds": [17], "arms": "sft,mix50", "train_time": "04:00:00",
    },
    "exec": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        # `coverage` joins `exec` here: both are small cells bounded by CRUXEval's 800 programs,
        # and both are execution-verified. `coverage`'s eval set is DexBench's, which is the one
        # external benchmark in the grid.
        "domains": ["exec", "coverage"], "seeds": list(SEEDS_SMALL), "arms": "full",
        "train_time": "04:00:00",
    },
    "large": {
        "models": ["llama31-8b", "gemma3-12b"],
        "domains": ["code", "mt_en-de", "sql", "d2t", "fmt", "algebra", "diacritics"],
        "seeds": list(SEEDS_LARGE), "arms": "core", "train_time": "12:00:00",
    },
    "relearn": {
        "models": ["llama32-3b", "llama31-8b"],
        "domains": ["code", "mt_en-de", "sql", "d2t", "fmt", "algebra"],
        "seeds": [17], "arms": "relearn", "train_time": "04:00:00",
    },
    "fullft": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["code", "mt_en-de", "sql", "d2t", "fmt"],
        "seeds": [17], "arms": "fullft", "train_time": "10:00:00",
    },
    "base_replicate": {   # RQ2's instruct-vs-pretrained moderator, one domain
        "models": ["llama31-8b-base"], "domains": ["mt_en-de"],
        "seeds": [17], "arms": "core", "train_time": "10:00:00",
    },
}

SMALL_PARTITIONS = ["a30", "h100", "h200"]
LARGE_PARTITIONS = ["h200", "h100"]


def size_gb(model: str) -> float:
    m = resolve_model(model)
    return {"olmo2-1b": 1.0, "llama32-3b": 3.0, "gemma3-4b": 4.0,
            "llama31-8b": 8.0, "gemma3-12b": 12.0}.get(model, float(m["hidden_size"]) / 512)



def resolvable_arms(arm_names: list[str], domain: str, model: str) -> tuple[list[str], list[str]]:
    """Drop dose rungs this domain's corpus is too small to express, and say which.

    A `mixN` arm reverses N % of the training pairs. For the rung to be a DOSE rather than
    noise, those reversed pairs have to survive contact with the optimizer: below one effective
    batch they can all land in a single step, and the arm becomes an expensive way to rerun
    `sft`. Measured against llama32-3b's effective batch of 64 (16 x 4):

        domain      train   1%    5%    10%   25%   50%
        mt_en-de     6500     65   325   650  1625  3250
        exec          302      3    15    30    76   151
        coverage      400      4    20    40   100   200

    `exec` and `coverage` are bounded by CRUXEval's 800 programs and `exec` lost a further 198
    to the execution audit (Amendment 15), so their bottom three rungs reverse 3-40 pairs. This
    is checked rather than hand-maintained: a domain whose corpus changes gets the right rungs
    without anyone remembering to edit a list, which is how `exec` came to be requesting a
    3-pair dose in the first place.

    Returns (kept, dropped). Non-dose arms are never dropped -- `sft`, `rev`, `flip`, `replay`
    and the rest do not have a dose to under-resolve.
    """
    import re

    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"][model]
    eff_batch = int(cfg["per_device_batch"]) * int(cfg["grad_accum"])
    try:
        n_train = sum(1 for _ in open(ROOT / "data" / domain / "train.jsonl"))
    except OSError:
        return arm_names, []                      # not built yet; the pack will say so

    kept, dropped = [], []
    for arm in arm_names:
        m = re.fullmatch(r"mix(\d+)", arm)
        if not m:
            kept.append(arm)
            continue
        reversed_pairs = round(n_train * int(m.group(1)) / 100)
        (kept if reversed_pairs >= eff_batch else dropped).append(arm)
    return kept, dropped


#: CPUs per job. 16 of a node's 64, which is modest for scheduling and matters for the
#: EXECUTION-SCORED domains: `exec_workers` is derived from the allocation (never hardcoded --
#: see bidir.domains._common.exec_workers), so 8 CPUs would cap `code`'s eval at 7 concurrent
#: programs for ~5,000 executions. Training does not execute anything; the eval does, and the
#: two share this setting because a cell is one pipeline.
CPUS = 16


def sub(name, argv, *, partition, time, dep=None, mem="64G", extra=None, dry=False,
        cpus=None) -> str | None:
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem, "--cpus", str(cpus or CPUS)]
    if dep:
        cmd += ["--dependency", f"afterok:{dep}"]
    cmd += list(extra or [])
    if dry:
        cmd.append("--dry-run")
    cmd += ["--argv"] + argv
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if dry:
        print(f"[dry] {name:<40s} {partition:<5s} {time}  {' '.join(argv[:6])} ...")
        return None
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith("submitted "):
            return line.split()[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--models", default=None, help="override the tier's model list")
    ap.add_argument("--domains", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max-submit", type=int, default=None, help="submit at most N cells (rest stay unsubmitted)")
    ap.add_argument("--eval-only", action="store_true", help="skip training; submit eval passes only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    t = TIERS[a.tier]
    models = (a.models.split(",") if a.models else t["models"])
    domains_ = (a.domains.split(",") if a.domains else t["domains"])
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else t["seeds"]
    arms_spec = t["arms"]
    arm_names = list(arm_registry.TIERS.get(arms_spec, ())) or [x for x in arms_spec.split(",")]
    units = arm_registry.units(arm_names)

    # Rungs this corpus cannot express are dropped per (domain, model), so a small cell
    # contributes the rungs it can resolve instead of four arms that all re-run `sft`.
    resolved: dict[tuple[str, str], list[str]] = {}
    for m in models:
        for d in domains_:
            keep, drop = resolvable_arms(arm_names, d, m)
            resolved[(d, m)] = keep
            if drop:
                print(f"  {d}/{m}: dropping {','.join(drop)} — fewer reversed pairs than one "
                      f"effective batch, so the rung cannot carry a dose")

    cells = [(d, m, s) for m in models for d in domains_ for s in seeds]
    # Budget from the arms each cell will ACTUALLY train, not from the tier's nominal list --
    # otherwise the estimate counts rungs the rung filter just dropped.
    total_units = sum(arm_registry.units(resolved[(d, m)]) for d, m, _ in cells)
    print(f"tier {a.tier}: {len(cells)} cells, "
          f"{min(len(v) for v in resolved.values())}-{max(len(v) for v in resolved.values())} "
          f"arms per cell = {total_units:.0f} adapter-units\n")
    if a.max_submit:
        cells = cells[: a.max_submit]

    n = 0
    for i, (domain, model, seed) in enumerate(cells):
        small = size_gb(model) <= 4.0
        parts = SMALL_PARTITIONS if small else LARGE_PARTITIONS
        partition = parts[i % len(parts)]
        extra = ["--exclude", "g-06-01"] if partition == "h100" and not small else []
        tag = f"{a.tier}"
        jid = None
        if not a.eval_only:
            cell_arms = ",".join(resolved[(domain, model)])
            jid = sub(f"tr_{domain}_{model}_s{seed}",
                      ["scripts/20_train_pack.py", "--domain", domain, "--model", model,
                       "--seed", str(seed), "--arms", cell_arms],
                      partition=partition, time=t["train_time"], extra=extra, dry=a.dry_run)
        ev_part = "h200" if not small else partition
        sub(f"ev_{domain}_{model}_s{seed}",
            ["-m", "bidir.evaluate", "--domain", domain, "--model", model, "--seed", str(seed),
             "--arms", ",".join(["base"] + arm_names), "--tag", tag],
            partition=ev_part, time="03:00:00", dep=jid, extra=extra, dry=a.dry_run)
        n += 1

    print(f"\n{n} cells {'planned' if a.dry_run else 'submitted'}. "
          f"Read them with: python scripts/50_contrasts.py --run <result dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
