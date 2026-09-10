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

_REPLAY_POOL_CACHE: dict[tuple, list] = {}


def replay_rows(n: int, seed: int, split: str = "train",
                target_lengths: Optional[Sequence[int]] = None) -> list[TrainRow]:
    """Generic single-turn instruction rows from the tulu-3 SFT mixture (cached in HF_HOME).

    LENGTH-MATCHED, not merely count-matched. `replay` exists to be an equal-sized substitution
    for the reversed pairs in `mix50`, so that `replay - sft` isolates ordinary forgetting and
    `mix50 - replay` isolates what direction specifically buys. Matching on row COUNT alone does
    not deliver that: measured 2026-09-10, an unmatched tulu-3 sample ran 1.24x the character
    budget of the mt_en-de pairs it replaced, which would have confounded both contrasts with a
    token-budget difference.

    So when `target_lengths` is given (the lengths of the pairs being replaced), each replay row
    is drawn to be the closest unused match to one of them. The realised ratio is reported by
    `scripts/16_audit_criteria.py` and lands in every run manifest via `direction_balance`.
    """
    from datasets import load_dataset

    cfg = load_config("domains/replay.yaml")
    max_chars = int(cfg.get("max_chars", 1500))
    # Cached: the pool is identical for every cell, and reloading a 40k-row dataset once per
    # (cell, arm) turned the budget audit into a 13-minute job.
    key = (cfg["hf_id"], cfg.get("hf_split"), max_chars)
    if key in _REPLAY_POOL_CACHE:
        pool = list(_REPLAY_POOL_CACHE[key])
    else:
        ds = load_dataset(cfg["hf_id"], split=cfg.get("hf_split", "train[:20000]"))
        pool = []
        for ex in ds:
            msgs = ex["messages"]
            if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
                continue
            u, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
            if not u or not a or len(u) + len(a) > max_chars:
                continue
            pool.append((ex["id"], u, a))
        _REPLAY_POOL_CACHE[key] = list(pool)
    rng = random.Random(seed + 7919)
    rng.shuffle(pool)
    if split == "val":
        pool = pool[::-1]
    if n > len(pool):
        raise ValueError(f"replay pool has {len(pool)} usable rows, need {n}; raise hf_split")

    if target_lengths is None:
        chosen = pool[:n]
    else:
        # Greedy nearest-length matching. Sorting both sides and walking them together would be
        # faster, but it correlates the choice with the pool's own ordering; matching each
        # target against the remaining pool keeps the draw independent of that.
        by_len = sorted(range(len(pool)), key=lambda i: len(pool[i][1]) + len(pool[i][2]))
        lens = [len(pool[i][1]) + len(pool[i][2]) for i in by_len]
        used: set[int] = set()
        chosen = []
        import bisect
        for t in list(target_lengths)[:n]:
            j = bisect.bisect_left(lens, t)
            best, bestd = None, None
            for k in range(max(0, j - 40), min(len(by_len), j + 40)):
                if by_len[k] in used:
                    continue
                d = abs(lens[k] - t)
                if bestd is None or d < bestd:
                    best, bestd = k, d
            if best is None:                      # exhausted the neighbourhood; take any unused
                best = next(k for k in range(len(by_len)) if by_len[k] not in used)
            used.add(by_len[best])
            chosen.append(pool[by_len[best]])
        while len(chosen) < n:                    # only if target_lengths was short
            k = next(k for k in range(len(by_len)) if by_len[k] not in used)
            used.add(by_len[k])
            chosen.append(pool[by_len[k]])

    out = []
    for rid, u, a in chosen:
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
        # Match on the RENDERED example, not the raw pair. The domain's own instruction wrapper
        # is most of the sequence for some cells — SQL prepends a whole schema, so its pairs are
        # 185 raw characters and 988 rendered ones. Targeting the raw length matched replay to a
        # quantity the trainer never sees and left the SQL arm at 0.80x.
        from bidir import domains as _domains
        from bidir import prompts as _prompts
        _mod = _domains.get(domain)

        def _rendered_len(pair) -> int:
            ex = _prompts.build_example({**pair.model_dump(), "task": "fwd"}, _mod)
            return (sum(len(m["content"]) for m in ex["prompt"])
                    + sum(len(m["content"]) for m in ex["completion"]))

        # A replay row is rendered under the same system prompt but WITHOUT the domain
        # instruction, so subtract that fixed overhead from the target it has to hit.
        _wrapper = len(_prompts.SYSTEM)
        target = [max(1, _rendered_len(p) - _wrapper) for p in replaced]
        rows = ([TrainRow.from_pair(p, "fwd") for p in keep]
                + replay_rows(len(replaced), seed, split, target_lengths=target))
    elif arm.mixed_task:
        cfg = load_config("domains/mixedtask.yaml")
        others = list(mixed_task_others or [d for d in cfg["domains"] if d != domain])
        if len(others) != int(cfg.get("n_others", 4)):
            raise ValueError(f"mixedtask needs {cfg.get('n_others', 4)} other domains, got {others}")
        rows = mixed_task_rows(domain, split, len(pairs), seed, float(cfg.get("share", 0.2)), others)
    else:
        for t in arm.tasks:
            if t in ("pos", "neg"):
                continue        # auxiliary pools are loaded below, from the domain
            rows += [TrainRow.from_pair(p, t) for p in pairs]

    if arm.aux_tasks:
        from bidir import domains as _domains
        mod = _domains.get(domain)
        if not hasattr(mod, "aux_pairs"):
            raise ValueError(f"arm {arm.name!r} needs auxiliary pools, which {domain} does not provide")
        for t in arm.aux_tasks:
            rows += [TrainRow.from_pair(p, t) for p in mod.aux_pairs(t, split)]

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
