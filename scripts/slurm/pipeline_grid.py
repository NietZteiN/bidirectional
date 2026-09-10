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
        "domains": ["code", "mt_en-de", "mt_de-en", "sql", "d2t", "fmt"],
        "seeds": list(SEEDS_SMALL), "arms": "full", "train_time": "10:00:00",
    },
    "small_zh": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["mt_en-zh", "mt_zh-en"],
        "seeds": list(SEEDS_SMALL), "arms": "core", "train_time": "06:00:00",
    },
    "ladder": {   # RQ3: the synthetic invertibility ladder
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["fmt_det75", "fmt_det50", "fmt_det25", "fmt_det00"],
        "seeds": [17], "arms": "sft,mix50", "train_time": "04:00:00",
    },
    "exec": {
        "models": ["llama32-3b", "gemma3-4b", "olmo2-1b"],
        "domains": ["exec"], "seeds": list(SEEDS_SMALL), "arms": "full", "train_time": "04:00:00",
    },
    "large": {
        "models": ["llama31-8b", "gemma3-12b"],
        "domains": ["code", "mt_en-de", "sql", "d2t", "fmt"],
        "seeds": list(SEEDS_LARGE), "arms": "core", "train_time": "12:00:00",
    },
    "relearn": {
        "models": ["llama32-3b", "llama31-8b"],
        "domains": ["code", "mt_en-de", "sql", "d2t", "fmt"],
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


def sub(name, argv, *, partition, time, dep=None, mem="64G", extra=None, dry=False) -> str | None:
    cmd = [sys.executable, str(SUBMIT), "--name", name, "--partition", partition,
           "--time", time, "--mem", mem]
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

    cells = [(d, m, s) for m in models for d in domains_ for s in seeds]
    print(f"tier {a.tier}: {len(cells)} cells x {len(arm_names)} arms "
          f"({units:.1f} adapter-units each) = {len(cells) * units:.0f} adapter-units\n")
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
            jid = sub(f"tr_{domain}_{model}_s{seed}",
                      ["scripts/20_train_pack.py", "--domain", domain, "--model", model,
                       "--seed", str(seed), "--arms", arms_spec],
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
