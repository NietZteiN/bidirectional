#!/usr/bin/env python
"""Train every arm of ONE (domain, model, seed) cell, back to back, in one SLURM job.

    python scripts/20_train_pack.py --domain mt_en-de --model llama32-3b --seed 17 --arms gate

This is the packing rule the whole plan rests on. The binding constraint on juno is not GPU
time but CONCURRENCY: the h200 partition QoS caps a user at 4 running jobs, shared with every
other project in this account. One job per adapter would leave 700 small-grid adapters queued
behind a 4-wide door; one job per cell turns that into ~60 jobs.

Two properties make it safe under a 2-day walltime:
  * an arm whose `final/` already exists is SKIPPED, so a job killed at walltime resumes by
    resubmission and costs only the arm it was in the middle of;
  * each arm is a fresh subprocess, so a CUDA OOM or a bad tokenizer in one arm cannot take
    the rest of the cell down with it — the pack reports which arms failed and exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as arm_registry  # noqa: E402
from bidir.config import GLOBAL_SEED, RUNS_DIR  # noqa: E402
from bidir.train import adapter_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--arms", default="gate", help="tier name (gate|core|full|relearn|fullft) or a comma list")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain arms whose adapter already exists")
    a = ap.parse_args()

    names = list(arm_registry.TIERS.get(a.arms, ())) or [x.strip() for x in a.arms.split(",") if x.strip()]
    names = [n for n in names if arm_registry.resolve(n).trains]
    # relearn-k continues from `sft`, so `sft` must be trained first if it is in the pack.
    names.sort(key=lambda n: (arm_registry.resolve(n).init_from is not None, names.index(n)))

    results: list[dict] = []
    t0 = time.time()
    for name in names:
        out = adapter_dir(a.domain, a.model, name, a.rank, a.seed,
                          root="adapters_fullft" if arm_registry.resolve(name).full_ft else "adapters")
        if (out / "final").exists() and not a.force:
            print(f"[pack] skip {name}: {out / 'final'} exists", flush=True)
            results.append({"arm": name, "status": "skipped"})
            continue
        cmd = [sys.executable, "-m", "bidir.train", "--domain", a.domain, "--arm", name,
               "--model", a.model, "--seed", str(a.seed)]
        if a.dry_run:
            cmd.append("--dry-run")
        print(f"[pack] === {name} === {' '.join(cmd)}", flush=True)
        t = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ,
                            "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}).returncode
        results.append({"arm": name, "status": "ok" if rc == 0 else "failed", "rc": rc,
                        "seconds": round(time.time() - t, 1)})
        print(f"[pack] {name}: rc={rc} in {results[-1]['seconds']}s", flush=True)

    failed = [r for r in results if r["status"] == "failed"]
    log_dir = RUNS_DIR / "packs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{a.domain}__{a.model}__s{a.seed}__{a.arms}.json").write_text(json.dumps({
        "domain": a.domain, "model": a.model, "seed": a.seed, "arms": names,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": round(time.time() - t0, 1), "results": results}, indent=2))
    print(f"[pack] done in {round(time.time() - t0, 1)}s: "
          f"{sum(1 for r in results if r['status'] == 'ok')} trained, "
          f"{sum(1 for r in results if r['status'] == 'skipped')} skipped, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
