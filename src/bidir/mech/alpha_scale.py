"""Experiment 2: scale the LoRA delta and watch both directions.

    python -m bidir.mech.alpha_scale --domain mt_en-de --model llama32-3b --arm sft \
        --alphas 0,0.125,0.25,0.5,0.75,1.0

A LoRA adapter contributes `(alpha/r) * B @ A` to each targeted weight. Scaling `lora_alpha`
scales that contribution continuously, so a = 0 is the base model and a = 1 the trained one,
with a well-defined path between them that costs no training.

What the shapes mean. If reverse success falls to zero by a = 0.25 while forward accuracy is
still climbing at a = 1, then whatever forward-only training installs destroys the inverse
long before it finishes learning the forward task — a cheap direction in weight space, which
is what H1 predicts. If instead the two curves move together, the loss is entangled with what
the model learned and H1 is in trouble.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import load_pairs
from bidir.train import adapter_dir


def scaled_adapter(src: Path, dst: Path, alpha_scale: float) -> Path:
    """Write a copy of the adapter whose `lora_alpha` is scaled by `alpha_scale`.

    Rewriting the config rather than the weights keeps the tensors byte-identical, so any
    difference along the curve is the scaling and not a re-quantized checkpoint. a = 0 is
    handled by the caller as "no adapter": PEFT rejects lora_alpha = 0.
    """
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    cfg_path = dst / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["lora_alpha"] = float(cfg["lora_alpha"]) * float(alpha_scale)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="sft")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alphas", default="0,0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    ensure_obtune()
    from bidir import engine as eng

    alphas = [float(x) for x in a.alphas.split(",")]
    dcfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    src = adapter_dir(a.domain, a.model, a.arm, a.rank, a.seed) / "final"
    if not src.exists():
        raise SystemExit(f"adapter missing: {src}")

    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"alpha_{a.domain}_{a.model}_{a.arm}"
    variants: list[tuple[float, Optional[str]]] = []
    for al in alphas:
        variants.append((al, None if al == 0.0 else str(scaled_adapter(src, scratch / f"a{al:g}", al))))

    e = eng.get_engine(a.model, {**dcfg.get("engine", {}), "max_loras": max(2, len(variants))})
    rows: list[dict] = []
    for direction in ("forward", "reverse"):
        msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
        for al, adapter in variants:
            raw, _ = eng.generate(e, msgs, [adapter] * len(msgs), dcfg.get("sampling", {}))
            outs = [prompts.extract_answer(r, "simple") for r in raw]
            scored = mod.score_batch(direction, outs, insts, dcfg)
            rows.append({"alpha": al, "direction": direction, "n": len(scored),
                         "strict": sum(r["strict"] for r in scored) / len(scored),
                         "echo": sum(r.get("echo", 0) for r in scored) / len(scored),
                         "off_target": sum(r.get("off_target", 0) for r in scored) / len(scored)})
            print(f"  a={al:<6g} {direction:<8s} strict={rows[-1]['strict']:.4f} "
                  f"echo={rows[-1]['echo']:.3f}", flush=True)

    out_dir = Path(a.out) if a.out else (RESULTS_DIR / "mech" / "alpha_scale" /
                                         f"{a.domain}__{a.model}__{a.arm}_s{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "curve.json").write_text(json.dumps(
        {"domain": a.domain, "model": a.model, "arm": a.arm, "seed": a.seed, "alphas": alphas,
         "n_instances": len(insts), "adapter": str(src),
         "finished_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2))
    print(f"[alpha] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
