#!/usr/bin/env python
"""Audit every domain's criterion and every arm's budget claim, before spending GPU-hours.

    python scripts/16_audit_criteria.py                    # everything
    python scripts/16_audit_criteria.py --domains sql,code

Two audits, both of which catch the same species of bug: a NOMINAL parameter that is not the
quantity it is claimed to be. The RQ3 ladder had one (drop share is not determinability) and it
put three of four rungs on the floor.

**1. Criterion audit.** Each domain's scorer is fed four oracle inputs and must behave:

    gold      the correct answer          -> strict = 1   (a criterion that cannot award a
                                                           perfect answer measures nothing)
    echo      the model's own input       -> strict = 0   (echo rose under forward-only
                                                           training; a criterion that lets it
                                                           through measures the criterion)
    empty     ""                          -> strict = 0
    garbage   unrelated text              -> strict = 0

Anything else is a broken criterion, and a broken criterion produces a full, plausible, wrong
table that no downstream analysis can detect.

**2. Budget audit.** Every arm except `flip` and `fwd2x` claims to be matched to `sft` on
instances, sequence tokens and optimizer steps. That claim is load-bearing for the paper's
headline — "reverse data is free" — and for the primary contrast `mix50 - replay`. Nominal
instance counts matching is not the same as token budgets matching, so this measures the
tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as arm_registry  # noqa: E402
from bidir import domains, prompts  # noqa: E402
from bidir.config import DATA_DIR, RESULTS_DIR, ensure_obtune, load_config, resolve_model  # noqa: E402
from bidir.mixture import build_mixture  # noqa: E402

GARBAGE = "The quick brown fox jumps over the lazy dog. This is not an answer to anything."

#: Criteria that route through a FROZEN SCORER MODEL — a whole vLLM engine — and so cannot be
#: audited without a GPU. That is itself worth knowing: these two criteria are the only ones in
#: the project that cannot be sanity-checked cheaply, which makes them the ones most worth
#: reading the numbers on when the gate first runs.
NEEDS_GPU = {("sql", "reverse"): "round-trip through a frozen NL->SQL parser",
             ("d2t", "forward"): "round-trip through a frozen triple extractor"}


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def audit_criterion(cell: str, n: int = 40) -> dict:
    mod = domains.get(cell)
    cfg = load_config(f"domains/{cell}.yaml")
    insts = [p.model_dump() for p in __import__("bidir.schema", fromlist=["read_pairs"])
             .read_pairs(DATA_DIR / cell / "test.jsonl")[:n]]
    out: dict = {"cell": cell, "n": len(insts), "directions": {}, "skipped": {}}
    have_gpu = _gpu_available()
    for direction in ("forward", "reverse"):
        why = NEEDS_GPU.get((cell, direction))
        if why and not have_gpu:
            out["skipped"][direction] = f"needs a GPU: {why}"
            continue
        gold = [prompts.completion_for(i, direction) for i in insts]
        echo = [prompts.input_for(i, direction) for i in insts]
        probes = {"gold": gold, "echo": echo, "empty": [""] * len(insts),
                  "garbage": [GARBAGE] * len(insts)}
        res = {}
        for name, outs in probes.items():
            try:
                scored = mod.score_batch(direction, outs, insts, cfg)
                res[name] = {
                    "strict": sum(r["strict"] for r in scored) / len(scored),
                    "echo_flag": sum(r.get("echo", 0) for r in scored) / len(scored),
                    "off_target": sum(r.get("off_target", 0) for r in scored) / len(scored),
                }
            except KeyError as e:
                # A threshold that is not frozen yet is the base gate's job, not a defect.
                msg = str(e)
                res[name] = ({"pending": "thresholds not frozen; run scripts/15_base_gate.py"}
                             if "not frozen" in msg else {"error": f"KeyError: {msg}"})
            except Exception as e:  # a scorer that raises on an oracle input IS a bug
                res[name] = {"error": f"{type(e).__name__}: {e}"}
        out["directions"][direction] = res
    return out


def verdict(a: dict) -> list[str]:
    """What is WRONG. Silence is a pass."""
    bad = []
    for direction, r in a["directions"].items():
        for name in ("gold", "echo", "empty", "garbage"):
            if "pending" in r.get(name, {}):
                continue
            if "error" in r.get(name, {}):
                bad.append(f"{a['cell']}/{direction}: {name} raised — {r[name]['error'][:90]}")
        g = r.get("gold", {})
        if "strict" in g and g["strict"] < 0.95:
            bad.append(f"{a['cell']}/{direction}: the GOLD answer only scores "
                       f"{g['strict']:.2f} — the criterion cannot award a correct answer")
        for name in ("echo", "empty", "garbage"):
            v = r.get(name, {})
            if "strict" in v and v["strict"] > 0.02:
                bad.append(f"{a['cell']}/{direction}: {name.upper()} scores {v['strict']:.2f} — "
                           f"the criterion awards a non-answer")
    return bad


def audit_budget(cell: str, model: str, arms_: list[str], n: int = 400) -> dict:
    """Sequence and supervised tokens per arm, as a ratio to `sft`."""
    ensure_obtune()
    from transformers import AutoTokenizer

    from obtune import prompts as oprompts
    from bidir.train import measure_lengths

    mcfg = resolve_model(model)
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = domains.get(cell)

    out: dict = {"cell": cell, "model": model, "arms": {}}
    for name in arms_:
        spec = arm_registry.resolve(name)
        rows = build_mixture(spec, cell, "train", seed=17)
        scale = len(rows) / max(1, min(n, len(rows)))   # extrapolate from a sample
        sample = rows[:n]
        ex = [oprompts.to_trl_example(prompts.build_example(r.model_dump(), mod), tok) for r in sample]
        _, stats = measure_lengths(ex, [r.task for r in sample], tok, int(mcfg["max_seq_len"]))
        out["arms"][name] = {
            "n_rows": len(rows),
            "sequence_tokens": stats["sequence_tokens_total"] * scale,
            "supervised_tokens": stats["supervised_tokens_total"] * scale,
            "mean_len": stats["len_mean"],
        }
    base = out["arms"].get("sft")
    if base:
        for name, v in out["arms"].items():
            v["rows_vs_sft"] = v["n_rows"] / base["n_rows"]
            v["seq_tokens_vs_sft"] = v["sequence_tokens"] / max(1.0, base["sequence_tokens"])
            v["sup_tokens_vs_sft"] = v["supervised_tokens"] / max(1.0, base["supervised_tokens"])
    return out


def budget_verdict(b: dict, tol: float = 0.10) -> list[str]:
    bad = []
    for name, v in b["arms"].items():
        spec = arm_registry.resolve(name)
        if spec.cost_units >= 2.0 or name == "sft":
            continue      # flip and fwd2x are the doubled references; they are meant to differ
        r = v.get("seq_tokens_vs_sft")
        if r is not None and abs(r - 1.0) > tol:
            bad.append(f"{b['cell']}/{name}: sequence tokens are {r:.2f}x sft "
                       f"(rows {v['rows_vs_sft']:.2f}x) — this arm claims to be budget-matched")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--model", default="llama32-3b")
    ap.add_argument("--budget-arms", default="sft,mix5,mix50,rev,replay,flip,fwd2x")
    ap.add_argument("--skip-budget", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    built = sorted(p.parent.name for p in DATA_DIR.glob("*/build_report.json"))
    cells = a.domains.split(",") if a.domains else built
    report: dict = {"criterion": [], "budget": [], "problems": []}

    if not _gpu_available():
        print("(no GPU here: the two frozen-scorer criteria are skipped and named below)\n")

    print("=== criterion audit: gold must pass, echo/empty/garbage must not ===")
    for cell in cells:
        if cell not in built:
            print(f"  {cell:<12s} not built, skipped")
            continue
        r = audit_criterion(cell, a.n)
        report["criterion"].append(r)
        bad = verdict(r)
        report["problems"] += bad
        def col(direction):
            if direction in r["skipped"]:
                return f"{direction[:3]} SKIPPED (needs GPU)"
            d = r["directions"][direction]
            if any("pending" in d.get(k, {}) for k in ("gold", "echo")):
                return f"{direction[:3]} PENDING (thresholds not frozen)"

            def g(k):
                return f"{d.get(k, {}).get('strict', float('nan')):.2f}"
            return f"{direction[:3]} gold={g('gold')} echo={g('echo')} garb={g('garbage')}"
        print(f"  {cell:<12s} {col('forward')} | {col('reverse')}"
              + ("   <-- PROBLEM" if bad else ""))

    if not a.skip_budget:
        print("\n=== budget audit: every matched arm must be ~1.00x sft in sequence tokens ===")
        arms_ = [x.strip() for x in a.budget_arms.split(",")]
        for cell in cells:
            if cell not in built:
                continue
            try:
                b = audit_budget(cell, a.model, arms_)
            except Exception as e:
                print(f"  {cell:<12s} FAILED: {type(e).__name__}: {e}")
                continue
            report["budget"].append(b)
            bad = budget_verdict(b)
            report["problems"] += bad
            cols = "  ".join(f"{n}={b['arms'][n]['seq_tokens_vs_sft']:.2f}x"
                             for n in arms_ if n in b["arms"] and n != "sft")
            print(f"  {cell:<12s} {cols}" + ("   <-- PROBLEM" if bad else ""))

    skipped = [(r["cell"], d, why) for r in report["criterion"] for d, why in r.get("skipped", {}).items()]
    if skipped:
        print("\n=== not audited (needs a GPU) ===")
        for cell, direction, why in skipped:
            print(f"  {cell}/{direction}: {why}")

    print("\n=== verdict ===")
    if report["problems"]:
        for p in report["problems"]:
            print(f"  ! {p}")
    else:
        print("  no problems found")

    out = RESULTS_DIR / "audit" / f"criteria_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[audit] wrote {out}")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
