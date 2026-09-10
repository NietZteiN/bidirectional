"""Mixture assembly: turns (arm, domain, split) into training rows.

The only construction that carries the paper is `split_directions`: a `mix*` arm REPLACES a
share of forward pairs with their reversal, partitioned by `pair_id`, so no pair is seen in
both directions (that would be `flip` with extra steps) and instance count, sequence tokens
and optimizer steps stay matched to `sft`. `replay` uses the same partition and puts generic
instruction rows where the reversed pairs would have gone, so `mix50 - replay` isolates what
direction buys over any equally sized substitution.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from bidir.arms import ArmSpec
from bidir.config import DATA_DIR, GLOBAL_SEED, load_config
from bidir.schema import PairInstance, TrainRow, read_pairs


def pairs_path(domain: str, split: str) -> Path:
    return DATA_DIR / domain / f"{split}.jsonl"


def load_pairs(domain: str, split: str) -> list[PairInstance]:
    p = pairs_path(domain, split)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run scripts/10_build_domain.py --domain {domain}")
    rows = read_pairs(p)
    rows.sort(key=lambda r: r.pair_id)  # file order is never load-bearing
    return rows


def split_directions(
    pairs: Sequence[PairInstance], reverse_fraction: float, seed: int
) -> tuple[list[PairInstance], list[PairInstance]]:
    """(forward_pairs, reversed_pairs), disjoint by pair_id, |reversed| = round(f * n)."""
    if not 0.0 <= reverse_fraction <= 1.0:
        raise ValueError(f"reverse_fraction must be in [0, 1], got {reverse_fraction}")
    ids = sorted({p.pair_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_rev = int(round(reverse_fraction * len(ids)))
    rev_ids = set(ids[:n_rev])
    fwd = [p for p in pairs if p.pair_id not in rev_ids]
    rev = [p for p in pairs if p.pair_id in rev_ids]
    return fwd, rev


def assert_direction_disjoint(rows: Sequence[TrainRow]) -> None:
    seen: dict[str, set[str]] = {}
    for r in rows:
        if r.task == "replay":
            continue
        seen.setdefault(r.pair_id, set()).add(r.task)
    both = [k for k, v in seen.items() if len(v) > 1]
    if both:
        raise AssertionError(f"{len(both)} pairs appear in both directions, e.g. {both[:3]}")


# --------------------------------------------------------------------------- replay rows

def replay_rows(n: int, seed: int, split: str = "train") -> list[TrainRow]:
    """Generic single-turn instruction rows from the tulu-3 SFT mixture (cached in HF_HOME).

    Single-turn only, under this project's shared system prompt, length-capped so a replay
    row costs about what a pair row costs. The sample is seeded and disjoint between train
    and val by construction (val draws from the tail of the shuffled pool).
    """
    from datasets import load_dataset

    cfg = load_config("domains/replay.yaml")
    ds = load_dataset(cfg["hf_id"], split=cfg.get("hf_split", "train[:20000]"))
    max_chars = int(cfg.get("max_chars", 1500))
    pool: list[tuple[str, str, str]] = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        u, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not u or not a or len(u) + len(a) > max_chars:
            continue
        pool.append((ex["id"], u, a))
    rng = random.Random(seed + 7919)
    rng.shuffle(pool)
    if split == "val":
        pool = pool[::-1]
    if n > len(pool):
        raise ValueError(f"replay pool has {len(pool)} usable rows, need {n}; raise hf_split")
    out = []
    for rid, u, a in pool[:n]:
        out.append(TrainRow(pair_id=f"replay::{rid}", domain="replay", subtask=str(cfg["hf_id"]),
                            side_a=u, side_b=a, split=split, task="replay"))
    return out


# --------------------------------------------------------------------------- mixed task

def mixed_task_rows(domain: str, split: str, n_total: int, seed: int, share: float,
                    others: Sequence[str]) -> list[TrainRow]:
    """This domain's forward pairs as `share` of a set filled equally by the other domains."""
    this = load_pairs(domain, split)
    rng = random.Random(seed + 104729)
    n_this = int(round(share * n_total))
    rng.shuffle(this)
    rows = [TrainRow.from_pair(p, "fwd") for p in this[:n_this]]
    per_other = (n_total - n_this) // max(1, len(others))
    for od in others:
        op = load_pairs(od, split)
        rng.shuffle(op)
        rows += [TrainRow.from_pair(p, "fwd") for p in op[:per_other]]
    return rows


# --------------------------------------------------------------------------- entry point

def build_mixture(arm: ArmSpec, domain: str, split: str, seed: int = GLOBAL_SEED,
                  mixed_task_others: Optional[Sequence[str]] = None) -> list[TrainRow]:
    pairs = load_pairs(domain, split)
    rows: list[TrainRow] = []

    if arm.relearn_k is not None:
        k = arm.relearn_k if split == "train" else max(8, arm.relearn_k // 10)
        rng = random.Random(seed + 15485863)
        pool = list(pairs)
        rng.shuffle(pool)
        rows = [TrainRow.from_pair(p, "rev") for p in pool[:k]]
    elif arm.reverse_fraction is not None:
        fwd, rev = split_directions(pairs, arm.reverse_fraction, seed)
        rows = [TrainRow.from_pair(p, "fwd") for p in fwd] + [TrainRow.from_pair(p, "rev") for p in rev]
    elif arm.replay_share is not None:
        keep, replaced = split_directions(pairs, arm.replay_share, seed)
        rows = [TrainRow.from_pair(p, "fwd") for p in keep] + replay_rows(len(replaced), seed, split)
    elif arm.mixed_task:
        cfg = load_config("domains/mixedtask.yaml")
        others = list(mixed_task_others or [d for d in cfg["domains"] if d != domain])
        if len(others) != int(cfg.get("n_others", 4)):
            raise ValueError(f"mixedtask needs {cfg.get('n_others', 4)} other domains, got {others}")
        rows = mixed_task_rows(domain, split, len(pairs), seed, float(cfg.get("share", 0.2)), others)
    else:
        for t in arm.tasks:
            rows += [TrainRow.from_pair(p, t) for p in pairs]

    assert_direction_disjoint(rows) if arm.reverse_fraction is not None else None
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def direction_balance(rows: Sequence[TrainRow]) -> dict[str, Any]:
    by_task = Counter(r.task for r in rows)
    n_pairs = len({r.pair_id for r in rows if r.task != "replay"})
    n_rev = by_task.get("rev", 0)
    n_fwd = by_task.get("fwd", 0)
    return {
        "n_rows": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "by_subtask": dict(sorted(Counter(r.subtask for r in rows if r.task != "replay").items())),
        "n_pairs": n_pairs,
        "reverse_share": n_rev / (n_fwd + n_rev) if (n_fwd + n_rev) else None,
        "replay_share": by_task.get("replay", 0) / len(rows) if rows else None,
    }
