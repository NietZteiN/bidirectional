"""Experiment 4: where in the stack does the collapse live?

    python -m bidir.mech.layer_ablate --domain mt_en-de --model llama32-3b --arm sft --groups 4

Zero the LoRA delta for one contiguous group of layers at a time and re-measure both
directions. A collapse concentrated in a few layers is a different claim from one spread
across the stack, and the paper should say which — and whether the answer is the same across
domains.

Zeroing rather than deleting: the adapter keeps its shape and every other layer's delta is
untouched, so the ablation is the only difference between runs. `lora_B` is the zeroed side
because `B` is initialized to zero at training start, so a zeroed `B` is exactly "this layer
was never adapted" and not an arbitrary perturbation.

Layer groups are FRACTIONS of depth, not absolute indices: the panel runs 16 to 48 layers, and
integer indices would silently probe different relative depths on different models (obtune hit
exactly this when Qwen's 28-layer indices were reused on CodeLlama's 32).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import load_pairs
from bidir.train import adapter_dir

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def ablated_adapter(src: Path, dst: Path, layers: set[int]) -> Path:
    """Copy the adapter with `lora_B` zeroed for the named layers."""
    from safetensors.torch import load_file, save_file
    import torch

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)
    for f in src.glob("*.safetensors"):
        tensors = load_file(str(f))
        for k in list(tensors):
            m = _LAYER_RE.search(k)
            if m and int(m.group(1)) in layers and "lora_B" in k:
                tensors[k] = torch.zeros_like(tensors[k])
        save_file(tensors, str(dst / f.name), metadata={"format": "pt"})
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="sft")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--groups", type=int, default=4, help="contiguous depth bands to ablate one at a time")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)

    ensure_obtune()
    from bidir import engine as eng

    mcfg = resolve_model(a.model)
    n_layers = int(mcfg["n_layers"])
    dcfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    src = adapter_dir(a.domain, a.model, a.arm, a.rank, a.seed) / "final"
    if not src.exists():
        raise SystemExit(f"adapter missing: {src}")

    bounds = [round(i * n_layers / a.groups) for i in range(a.groups + 1)]
    bands = [(bounds[i], bounds[i + 1]) for i in range(a.groups)]
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"ablate_{a.domain}_{a.model}_{a.arm}"

    variants: list[tuple[str, Optional[str]]] = [("none", str(src)), ("all", None)]
    for lo, hi in bands:
        variants.append((f"{lo}-{hi - 1}", str(ablated_adapter(src, scratch / f"L{lo}_{hi}", set(range(lo, hi))))))

    e = eng.get_engine(a.model, {**dcfg.get("engine", {}), "max_loras": max(2, len(variants))})
    rows = []
    for direction in ("forward", "reverse"):
        msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
        for label, adapter in variants:
            raw, _ = eng.generate(e, msgs, [adapter] * len(msgs), dcfg.get("sampling", {}))
            scored = mod.score_batch(direction, [prompts.extract_answer(r) for r in raw], insts, dcfg)
            rows.append({"ablated": label, "direction": direction,
                         "strict": sum(r["strict"] for r in scored) / len(scored),
                         "echo": sum(r.get("echo", 0) for r in scored) / len(scored)})
            print(f"  ablate={label:<8s} {direction:<8s} strict={rows[-1]['strict']:.4f}", flush=True)

    out_dir = RESULTS_DIR / "mech" / "layer_ablate" / f"{a.domain}__{a.model}__{a.arm}_s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(
        {"domain": a.domain, "model": a.model, "arm": a.arm, "n_layers": n_layers,
         "bands": [f"{lo}-{hi - 1}" for lo, hi in bands], "n_instances": len(insts),
         "note": "'none' = the full adapter, 'all' = the untouched base model",
         "finished_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2))
    print(f"[ablate] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
