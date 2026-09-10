"""Mechanism experiment 7: spectral repair of the fine-tuning update.

    python -m bidir.mech.spectral --domain mt_en-de --model llama32-3b --arm sft \
        --taus 0.5,0.75,1.0,1.5,2.0

Following "Spectral Unforgetting" (arXiv:2605.20296, `papers/spectral2026unforgetting.pdf`):
treat the fine-tuning update as low-rank task signal embedded in an IID-like noise residual,
apply the Donoho-Gavish optimal hard singular-value threshold, keep the structured high-energy
part and discard the spectral bulk. Closed form, no data, no retraining.

WHY THIS IS THE SHARPEST RQ5 INSTRUMENT WE HAVE. Adapter scaling (experiment 2) shrinks the
*whole* update uniformly, so a reverse-capability recovery under it is ambiguous between "the
collapse is a cheap direction in weight space" and "less fine-tuning is simply less damage".
Spectral repair removes a *component* of the update while keeping the part that carries the
forward task. If reverse success rises while forward success stays inside the seed band, the
capability was never gone — it was masked by a removable residue of the update. That is the
suppression hypothesis stated as an intervention rather than an inference.

THE CAVEAT THIS MODULE MUST NOT HIDE. DG-Hard was designed for FULL fine-tuning deltas, which
are full-rank matrices holding a low-rank signal plus a noise bulk. A LoRA delta is **already**
rank-r by construction: there is no bulk to remove, only r singular values to threshold among.
So on a LoRA arm this is a weaker instrument than the source paper's, and a null result on LoRA
is much weaker evidence than a null on a full fine-tuning delta — which is the arm this
experiment most wants.

**Full fine-tuning deltas are NOT yet supported.** Repairing one means loading both the base and
the tuned checkpoint and taking an SVD of every weight matrix, which is a different and much
heavier code path than rewriting an adapter. The `fullft_*` arms therefore raise here rather
than being silently handled by the LoRA path, which would compute a meaningless delta. Every
result file records `delta_is_lowrank_by_construction` so the caveat travels with the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bidir import domains, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import load_pairs
from bidir.train import adapter_dir


def omega(beta: float) -> float:
    """Donoho-Gavish coefficient for an unknown noise level (Gavish & Donoho 2014).

    tau = omega(beta) * median(singular values), with beta = min(m,n)/max(m,n). The polynomial
    is their published approximation; at beta = 1 it gives the familiar 2.858.
    """
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def dg_hard(delta, tau_scale: float = 1.0, max_rank: Optional[int] = None):
    """Hard-threshold the singular values of `delta`. Returns (repaired, report).

    `max_rank` caps how many components are considered at all, and for a LoRA delta it MUST be
    set to the adapter's own rank. `B @ A` has rank at most r by construction, so every
    singular value beyond the r-th is bf16 round-trip noise — and without the cap the median
    lands in that noise, the threshold collapses toward zero, and the "repaired" adapter comes
    out at HIGHER rank than the original with the numerical noise materialized as real
    components. Measured on a synthetic rank-8 adapter before the cap was added: rank 8 in,
    rank 64 out.
    """
    import torch

    m, n = delta.shape
    beta = min(m, n) / max(m, n)
    # float32: the threshold is a median over singular values and bf16 quantisation moves it.
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    if max_rank is not None:
        S = S[:max_rank]
        U, Vh = U[:, :max_rank], Vh[:max_rank, :]
    tau = omega(beta) * float(S.median()) * float(tau_scale)
    keep = int((S > tau).sum())
    S_cut = S.clone()
    S_cut[keep:] = 0.0
    repaired = (U @ torch.diag(S_cut) @ Vh).to(delta.dtype)
    return repaired, {"rank_considered": int(S.numel()), "rank_kept": keep,
                      "tau": tau, "beta": beta,
                      "energy_kept": float((S_cut ** 2).sum() / max(1e-12, (S ** 2).sum()))}


def repair_lora_adapter(src: Path, dst: Path, tau_scale: float) -> dict[str, Any]:
    """Rebuild a LoRA adapter from the spectrally filtered delta.

    The delta of a LoRA pair is (alpha/r) * B @ A. We threshold that product and re-factor the
    result back into a B'/A' pair by truncated SVD, so the output is a valid adapter of lower
    rank that vLLM and PEFT load unchanged. `alpha` is set to the new rank so the scaling
    factor stays 1 and the reconstruction is exact.
    """
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / r

    files = list(src.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {src}")
    tensors = load_file(str(files[0]))

    pairs: dict[str, dict[str, str]] = {}
    for k in tensors:
        if ".lora_A" in k:
            pairs.setdefault(k.split(".lora_A")[0], {})["A"] = k
        elif ".lora_B" in k:
            pairs.setdefault(k.split(".lora_B")[0], {})["B"] = k

    out: dict[str, Any] = {}
    reports, new_rank = [], 0
    for base, kv in sorted(pairs.items()):
        if "A" not in kv or "B" not in kv:
            continue
        A, B = tensors[kv["A"]], tensors[kv["B"]]
        delta = (B.float() @ A.float()) * scale
        # Cap at the adapter's own rank: see dg_hard's docstring.
        repaired, rep = dg_hard(delta, tau_scale, max_rank=r)
        rep["module"] = base
        reports.append(rep)
        k = max(1, rep["rank_kept"])          # rank 0 would be an empty adapter, not an adapter
        new_rank = max(new_rank, k)
        U, S, Vh = torch.linalg.svd(repaired.float(), full_matrices=False)
        root = torch.diag(S[:k].sqrt())
        out[kv["B"]] = (U[:, :k] @ root).to(B.dtype)
        out[kv["A"]] = (root @ Vh[:k, :]).to(A.dtype)

    # Pad every pair to one common rank: PEFT stores a single `r` per adapter.
    for base, kv in sorted(pairs.items()):
        if "A" not in kv or "B" not in kv:
            continue
        Bn, An = out[kv["B"]], out[kv["A"]]
        if Bn.shape[1] < new_rank:
            padB = torch.zeros(Bn.shape[0], new_rank - Bn.shape[1], dtype=Bn.dtype)
            padA = torch.zeros(new_rank - An.shape[0], An.shape[1], dtype=An.dtype)
            out[kv["B"]] = torch.cat([Bn, padB], dim=1)
            out[kv["A"]] = torch.cat([An, padA], dim=0)

    for k, v in tensors.items():
        out.setdefault(k, v)               # anything that is not a LoRA pair passes through

    save_file(out, str(dst / files[0].name), metadata={"format": "pt"})
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)
    cfg["r"] = new_rank
    cfg["lora_alpha"] = new_rank           # scale = 1: the factorization above is already scaled
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    kept = sum(r_["rank_kept"] for r_ in reports)
    before = sum(r_["rank_considered"] for r_ in reports)
    return {"tau_scale": tau_scale, "n_modules": len(reports), "new_rank": new_rank,
            "total_rank_considered": before, "total_rank_kept": kept,
            "mean_energy_kept": sum(r_["energy_kept"] for r_ in reports) / max(1, len(reports)),
            "per_module": reports[:8]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="sft", help="LoRA arms only; see the module docstring on full fine-tuning")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--taus", default="0.5,0.75,1.0,1.5,2.0",
                    help="multipliers on the Donoho-Gavish threshold; 1.0 is the published value")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)

    ensure_obtune()
    from bidir import arms as arm_registry
    from bidir import engine as eng

    spec = arm_registry.resolve(a.arm)
    if spec.full_ft:
        raise SystemExit(
            f"arm {a.arm!r} is a full fine-tune. Repairing that delta needs both checkpoints and an "
            "SVD of every weight matrix — a different code path from rewriting an adapter, and not "
            "implemented. Running it through the LoRA path would compute a meaningless delta.")
    root = "adapters"
    src = adapter_dir(a.domain, a.model, a.arm, a.rank, a.seed, root=root) / "final"
    if not src.exists():
        raise SystemExit(f"adapter missing: {src}")

    dcfg = load_config(f"domains/{a.domain}.yaml")
    mod = domains.get(a.domain)
    insts = [p.model_dump() for p in load_pairs(a.domain, "test")][: a.limit]
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"spectral_{a.domain}_{a.model}_{a.arm}"

    variants: list[tuple[str, Optional[str], dict]] = [("base", None, {}), ("unrepaired", str(src), {})]
    for t in [float(x) for x in a.taus.split(",")]:
        d = scratch / f"tau{t:g}"
        rep = repair_lora_adapter(src, d, t)
        variants.append((f"dghard_tau{t:g}", str(d), rep))
        print(f"  tau x{t:g}: rank {rep['total_rank_before']} -> {rep['total_rank_kept']} "
              f"across {rep['n_modules']} modules, energy kept {rep['mean_energy_kept']:.3f}", flush=True)

    e = eng.get_engine(a.model, {**dcfg.get("engine", {}), "max_loras": max(2, len(variants)),
                                 "max_lora_rank": 128})
    rows = []
    for direction in ("forward", "reverse"):
        msgs = [prompts.build_messages(i, direction, mod, "simple") for i in insts]
        for label, adapter, rep in variants:
            raw, _ = eng.generate(e, msgs, [adapter] * len(msgs), dcfg.get("sampling", {}))
            scored = mod.score_batch(direction, [prompts.extract_answer(r) for r in raw], insts, dcfg)
            rows.append({"variant": label, "direction": direction,
                         "strict": sum(x["strict"] for x in scored) / len(scored),
                         "echo": sum(x.get("echo", 0) for x in scored) / len(scored),
                         **({"repair": rep} if rep else {})})
            print(f"  {label:<16s} {direction:<8s} strict={rows[-1]['strict']:.4f}", flush=True)

    out_dir = RESULTS_DIR / "mech" / "spectral" / f"{a.domain}__{a.model}__{a.arm}_s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectral.json").write_text(json.dumps({
        "domain": a.domain, "model": a.model, "arm": a.arm, "seed": a.seed,
        "n_instances": len(insts), "adapter": str(src),
        # The caveat, carried in the data rather than only in the docstring.
        "delta_is_lowrank_by_construction": not spec.full_ft,
        "instrument_note": ("A LoRA delta is already rank-r, so there is no noise bulk to remove "
                            "and this is a weaker instrument than the source paper's. A null here "
                            "is much weaker evidence than a null on fullft_sft."),
        "hypothesis": ("suppression predicts reverse strict success rises above `unrepaired` "
                       "while forward stays inside the seed band"),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2))
    print(f"[spectral] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
