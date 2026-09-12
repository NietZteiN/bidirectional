"""One evaluation pass over a cell: every arm, both directions, one engine.

    python -m bidir.evaluate --domain mt_en-de --model llama32-3b --seed 17 \
        --arms base,sft,mix5,mix50,rev

Order matters and is deliberate (obtune's lesson, 2026-08-11): `trials.jsonl` is written
BEFORE the adapter-effectiveness guard runs. A run that trips the guard used to discard an
hour of generation and leave only a traceback, which made the failure undiagnosable. The
guard must still fail the job — the cells are not trustworthy — but the evidence has to
survive it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from bidir import arms as arm_registry
from bidir import domains, engine as eng, prompts
from bidir.config import GLOBAL_SEED, RESULTS_DIR, ensure_obtune, load_config, resolve_model
from bidir.mixture import load_pairs
from bidir.train import adapter_dir

DIRECTIONS = ("forward", "reverse")


#: An arm whose outputs match its reference at or above this rate did not take effect.
#: 0.999 rather than 1.0 -- see the guard in main() for why exact equality is not safe.
ADAPTER_IDENTICAL_MAX = 0.999


def resolve_systems(domain: str, model: str, arm_names: Sequence[str], seed: int,
                    rank: int = 32) -> dict[str, Optional[str]]:
    """arm -> adapter path (None = the untouched base model).

    `base` is REQUIRED in every pass: it is the reference every contrast is taken against and
    the only way to tell "fine-tuning destroyed the reverse direction" from "it was never
    there" (plan §3 RQ1).
    """
    systems: dict[str, Optional[str]] = {}
    for name in arm_names:
        spec = arm_registry.resolve(name)
        if not spec.trains:
            systems[name] = None
            continue
        root = "adapters_fullft" if spec.full_ft else "adapters"
        p = adapter_dir(domain, model, name, rank, seed, root=root) / "final"
        if not p.exists():
            raise FileNotFoundError(f"adapter missing for arm {name!r}: {p}")
        systems[name] = str(p)
    if "base" not in systems:
        systems = {"base": None, **systems}
    return systems


def build_requests(insts: Sequence[Mapping[str, Any]], systems: Mapping[str, Optional[str]],
                   directions: Sequence[str], strategies: Sequence[str], domain_mod: Any,
                   shots: Optional[Sequence[Mapping[str, Any]]] = None) -> list[dict[str, Any]]:
    from bidir.domains._common import cluster_key as _default_cluster

    cluster_of = getattr(domain_mod, "cluster_key", _default_cluster)
    reqs: list[dict[str, Any]] = []
    for sys_name, adapter in systems.items():
        for direction in directions:
            # Forward is only ever scored under `simple`: the elicitation ladder is a
            # reverse-direction question (does prompting rescue what tuning removed?), and
            # running it forward would quadruple the pass for nothing.
            strats = strategies if direction == "reverse" else ("simple",)
            for strategy in strats:
                for inst in insts:
                    reqs.append({
                        "trial_id": f"{sys_name}::{direction}::{strategy}::{inst['pair_id']}",
                        "system": sys_name, "direction": direction, "strategy": strategy,
                        "pair_id": inst["pair_id"], "subtask": inst["subtask"],
                        # The INDEPENDENT unit the bootstrap must resample. Equals pair_id
                        # everywhere except `code`, whose five obfuscation conditions share one
                        # source program -- 2,060 rows are 412 programs. Written per trial so a
                        # contrast never has to re-derive it from a domain module.
                        "cluster_id": cluster_of(inst),
                        "adapter": adapter, "inst": inst,
                        "messages": prompts.build_messages(inst, direction, domain_mod,
                                                           strategy=strategy, shots=shots),
                    })
    return reqs


def summarize(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    """Cell means keyed by (system, direction, strategy, subtask), plus an ALL row per cell."""
    cells: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["system"], r["direction"], r["strategy"], r["subtask"])].append(r)
        cells[(r["system"], r["direction"], r["strategy"], "ALL")].append(r)
    out = []
    for (system, direction, strategy, subtask), group in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        cell: dict[str, Any] = {"system": system, "direction": direction, "strategy": strategy,
                                "subtask": subtask, "n": len(group),
                                "n_pairs": len({r["pair_id"] for r in group})}
        for m in metrics:
            vals = [r[m] for r in group if isinstance(r.get(m), (int, float))]
            if vals:
                cell[m] = sum(vals) / len(vals)
        out.append(cell)
    return {"cells": out}


def adapter_effectiveness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every trained system must differ from its STARTING POINT, which is not always `base`.

    An arm continued from another adapter (`relearn*` starts from `sft`) differs from `base` no
    matter what, so a base-only check passes it even when it trained not at all — and "no
    recovery at small k" is exactly what erasure looks like. Each arm is therefore compared
    against the adapter it was initialised from, when it has one.
    """
    by_key: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_key[(r["direction"], r["strategy"], r["pair_id"])][r["system"]] = r["output_raw"]
    report: dict[str, Any] = {}
    for name in sorted({r["system"] for r in rows} - {"base"}):
        try:
            reference = arm_registry.resolve(name).init_from or "base"
        except KeyError:
            reference = "base"
        if reference not in {r["system"] for r in rows}:
            reference = "base"
        pairs = [(v[reference], v[name]) for v in by_key.values() if reference in v and name in v]
        if not pairs:
            continue
        identical = sum(1 for a, b in pairs if a == b)
        report[name] = {"reference": reference, "n_compared": len(pairs),
                        "identical_to_reference": identical,
                        "identical_rate": identical / len(pairs)}
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--arms", default="base,sft,mix5,mix50,rev", help="comma-separated, or a tier name")
    ap.add_argument("--rank", type=int, default=32,
                    help="LoRA r of the adapters to score. The adapter PATH encodes the rank, so "
                         "this must match what 20_train_pack.py trained or the adapters are not "
                         "found at all.")
    ap.add_argument("--strategies", default="simple", help="reverse-direction elicitation ladder")
    ap.add_argument("--limit", type=int, default=None, help="cap eval instances (default: the domain config's n_test)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="names the result directory; defaults to the arm set")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_obtune()
    arm_names = list(arm_registry.TIERS.get(args.arms, ())) or [a.strip() for a in args.arms.split(",") if a.strip()]
    if "base" not in arm_names:
        arm_names = ["base"] + arm_names
    dcfg = load_config(f"domains/{args.domain}.yaml")
    domain_mod = domains.get(args.domain)
    systems = resolve_systems(args.domain, args.model, arm_names, args.seed, rank=args.rank)

    insts = [p.model_dump() for p in load_pairs(args.domain, args.split)]
    if args.limit:
        insts = insts[: args.limit]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    shots = None
    if "few_shot" in strategies:
        # DEMONSTRATIONS COME FROM `val`, NOT `train`.
        #
        # Never the eval set, obviously. But `train` is wrong too, and wrong in a way that
        # biases the elicitation ladder specifically. Every `mix*` arm reverses a share of the
        # train pairs PARTITIONED BY pair_id, so at mix50 each demo has roughly a coin-flip
        # chance of being a pair that arm saw in reverse during training, while for `sft` it
        # never was. The ladder asks "does prompting rescue what tuning removed?" -- and the
        # answer would have been read off prompts whose demonstrations were memorised reverse
        # examples for the high-dose arms and unseen ones for `sft`. That is an arm-dependent
        # difference in the PROMPT, which is precisely what CLAUDE.md §3.3 exists to forbid.
        #
        # `val` is clean for this: it reaches the trainer only as `eval_dataset`, so no gradient
        # ever sees it, and `10_build_domain.py` asserts it disjoint from `test`. Deterministic
        # slice, so every arm and every instance gets byte-identical demonstrations.
        shots = [p.model_dump() for p in load_pairs(args.domain, "val")[: int(dcfg.get("n_shots", 2))]]

    reqs = build_requests(insts, systems, DIRECTIONS, strategies, domain_mod, shots)
    print(f"[bidir.eval] {len(insts)} instances x {len(systems)} systems -> {len(reqs)} generations", flush=True)

    e = eng.get_engine(args.model, dcfg.get("engine", {}))
    raw, ntok = eng.generate(e, [r["messages"] for r in reqs], [r["adapter"] for r in reqs],
                             dcfg.get("sampling", {}))

    # Score per (direction, strategy) group: every domain's scorer is a batch call.
    rows: list[dict[str, Any]] = [None] * len(reqs)  # type: ignore[list-item]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(reqs):
        groups[(r["direction"], r["strategy"])].append(i)
    for (direction, strategy), idx in sorted(groups.items()):
        outs = [prompts.extract_answer(raw[i], strategy) for i in idx]
        scored = domain_mod.score_batch(direction, outs, [reqs[i]["inst"] for i in idx], dcfg)
        for i, out, sc in zip(idx, outs, scored):
            r = reqs[i]
            rows[i] = {"trial_id": r["trial_id"], "system": r["system"], "direction": direction,
                       "strategy": strategy, "pair_id": r["pair_id"], "subtask": r["subtask"],
                       "adapter": r["adapter"], "output_raw": raw[i], "output": out,
                       "n_gen_tokens": int(ntok[i]), **sc}

    tag = args.tag or "-".join(a for a in arm_names if a != "base")
    out_dir = Path(args.out) if args.out else (RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
                                               / args.domain / args.model / f"{tag}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trials.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "inst"}, ensure_ascii=False) + "\n")

    metrics = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "n_gen_tokens"})
    summary = summarize(rows, metrics)
    effect = adapter_effectiveness(rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "domain": args.domain, "model": args.model, "seed": args.seed, "arms": arm_names,
        "systems": systems, "n_instances": len(insts), "strategies": strategies,
        "engine_version": e.version(), "adapter_effectiveness": effect,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        **prompts.provenance_block(), **summary}, indent=2))

    # ALWAYS PRINTED, so a 0.95 is visible even where it is not fatal.
    for name, rep in sorted(effect.items()):
        print(f"  [effect] {name:<12s} vs {rep['reference']:<8s} "
              f"identical={rep['identical_rate']:.4f} ({rep['identical_to_reference']}/"
              f"{rep['n_compared']})", flush=True)

    # The threshold is 0.999, not 1.0. An adapter that failed to load runs on base weights and
    # SHOULD give byte-identical text, but greedy decoding is not bitwise reproducible even for
    # identical requests -- two passes over the same 200 prompts on one engine moved tau by
    # 0.0010 COMET on 2026-09-12. Demanding EXACT equality therefore lets a failed adapter
    # through at 0.9995, and the result is the worst kind of table: a perfect copy of the base,
    # row for row, with nothing raised.
    #
    # Not lower than 0.999, because some arms are legitimately near-identical to their
    # reference: `relearn10` continues from `sft` and takes 40 steps on 10 examples, and a
    # deterministic format domain can have base and `sft` emit the same canonical string. Those
    # are real measurements, so the band is deliberately narrow and the rate is printed above.
    for name, rep in sorted(effect.items()):
        if rep["identical_rate"] >= ADAPTER_IDENTICAL_MAX:
            raise RuntimeError(
                f"system {name!r} produced output identical to {rep['reference']!r} on "
                f"{rep['identical_to_reference']}/{rep['n_compared']} trials "
                f"({rep['identical_rate']:.4f} >= {ADAPTER_IDENTICAL_MAX}) — it did not take "
                f"effect. Adapter was {systems.get(name)!r}. Rows are in "
                f"{out_dir}/trials.jsonl")

    for c in summary["cells"]:
        if c["subtask"] == "ALL" and c["strategy"] == "simple":
            print(f"  {c['system']:<12s} {c['direction']:<8s} strict={c.get('strict', float('nan')):.4f} "
                  f"echo={c.get('echo', float('nan')):.3f} n={c['n']}", flush=True)
    print(f"[bidir.eval] wrote {out_dir}", flush=True)

    # Mirror the small irreplaceable artifacts onto /work straight away. $BIDIR_OUT is scratch,
    # no retention policy for it has been confirmed, and trials.jsonl is the one output that
    # cannot be regenerated exactly -- greedy decoding is only approximately repeatable across
    # passes. Best effort: a failed mirror must never fail an eval that succeeded.
    try:
        import subprocess

        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "94_archive_trials.py")],
                       capture_output=True, timeout=300)
    except Exception as exc:                            # noqa: BLE001
        print(f"[bidir.eval] archive step skipped: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    # shutdown_and_exit, not sys.exit: vLLM's engine-core child does not always come back, and
    # job 391263 sat RUNNING for 6m14s on an H200 after printing its last line. Results are
    # already on disk by here; the exit code still reaches the sbatch template's trap.
    eng.shutdown_and_exit(main())
