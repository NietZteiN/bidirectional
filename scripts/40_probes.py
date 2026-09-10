#!/usr/bin/env python
"""General-ability probes for a set of arms, through lm-evaluation-harness on vLLM.

    python scripts/40_probes.py --model llama32-3b --cells mt_en-de,sql --arms base,sft,mix5 --tasks ifeval,gsm8k

These are the controls that separate DIRECTIONAL loss from ordinary degradation. RQ1's claim
is that the reverse capability falls to near zero while general ability falls a few points; a
paper that does not measure the second half has not made the claim.

IFEval carries a second job: it is direction-specified instruction following on an unrelated
task, so it separates "the model lost the inverse mapping" from "the model stopped following
instructions at all".

OVERLAP IS CHECKED BEFORE THE PROBE RUNS, not after. In the workshop paper 74 of HumanEval+'s
164 problems turned out to be in the training split, which made forward-only tuning look free;
the fix was a held-out probe, and the lesson was to write the overlap number down first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as arm_registry  # noqa: E402
from bidir.config import GLOBAL_SEED, RESULTS_DIR, resolve_model  # noqa: E402
from bidir.domains._common import normalize  # noqa: E402
from bidir.mixture import load_pairs  # noqa: E402
from bidir.train import adapter_dir  # noqa: E402

TASKS = {"ifeval": "ifeval", "gsm8k": "gsm8k", "mmlu": "mmlu", "mmlu_pro": "mmlu_pro"}
FEWSHOT = {"ifeval": 0, "gsm8k": 5, "mmlu": 5, "mmlu_pro": 5}


def overlap_check(cells: list[str], task: str) -> dict:
    """Count training strings that appear verbatim in the probe. Cheap, and it is the number
    the paper has to be able to quote."""
    try:
        from datasets import load_dataset
        if task == "gsm8k":
            probe = {normalize(r["question"]) for r in load_dataset("openai/gsm8k", "main", split="test")}
        elif task == "ifeval":
            probe = {normalize(r["prompt"]) for r in load_dataset("google/IFEval", split="train")}
        else:
            return {"checked": False, "reason": f"no overlap check implemented for {task}"}
    except Exception as e:
        return {"checked": False, "reason": f"{type(e).__name__}: {e}"}
    hits = {}
    for cell in cells:
        train = {normalize(p.side_a) for p in load_pairs(cell, "train")} | \
                {normalize(p.side_b) for p in load_pairs(cell, "train")}
        hits[cell] = len(train & probe)
    return {"checked": True, "probe_size": len(probe), "verbatim_overlap": hits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", required=True, help="comma list; adapters are looked up per cell")
    ap.add_argument("--arms", default="base,sft,mix5")
    ap.add_argument("--tasks", default="ifeval,gsm8k")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="instances per task (smoke tests)")
    a = ap.parse_args()

    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    arm_names = [x.strip() for x in a.arms.split(",") if x.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    hf_id = resolve_model(a.model)["hf_id"]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_root = RESULTS_DIR / "probes" / stamp / a.model
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "overlap.json").write_text(json.dumps(
        {t: overlap_check(cells, t) for t in tasks}, indent=2))

    jobs: list[tuple[str, str | None]] = []
    for cell in cells:
        for name in arm_names:
            spec = arm_registry.resolve(name)
            if not spec.trains:
                continue
            p = adapter_dir(cell, a.model, name, a.rank, a.seed,
                            root="adapters_fullft" if spec.full_ft else "adapters") / "final"
            if p.exists():
                jobs.append((f"{cell}__{name}", str(p)))
    if "base" in arm_names:
        jobs.insert(0, ("base", None))   # the untouched model is probed once, not once per cell

    results = {}
    for label, adapter in jobs:
        model_args = [f"pretrained={hf_id}", "dtype=bfloat16", "gpu_memory_utilization=0.80",
                      "max_model_len=4096"]
        if adapter:
            model_args += [f"lora_local_path={adapter}", "enable_lora=True", "max_lora_rank=64"]
        out_path = out_root / f"{label}.json"
        cmd = ["lm_eval", "--model", "vllm", "--model_args", ",".join(model_args),
               "--tasks", ",".join(TASKS[t] for t in tasks), "--batch_size", "auto",
               "--seed", str(a.seed), "--output_path", str(out_path)]
        for t in tasks:
            if FEWSHOT.get(t):
                cmd += ["--num_fewshot", str(FEWSHOT[t])]
                break
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print(f"[probes] === {label} === {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        results[label] = {"rc": rc, "output": str(out_path), "adapter": adapter}

    (out_root / "index.json").write_text(json.dumps(
        {"model": a.model, "cells": cells, "arms": arm_names, "tasks": tasks,
         "finished_utc": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2))
    failed = [k for k, v in results.items() if v["rc"] != 0]
    print(f"[probes] wrote {out_root}; {len(results) - len(failed)} ok, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
