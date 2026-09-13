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
    pairs: Sequence[PairInstance], reverse_fraction: float, seed: int,
    domain: Optional[str] = None,
) -> tuple[list[PairInstance], list[PairInstance]]:
    """(forward_pairs, reversed_pairs), disjoint by CLUSTER, |reversed| ~= round(f * n).

    Partitioned by the domain's `cluster_key`, which is `pair_id` everywhere except `code`,
    where five obfuscation conditions share one source program. Partitioning `code` by row --
    which is what this did -- put program P's condition L1b in the forward half and its L2 in
    the reverse half, and both sides of both rows derive from P's own code. The model then sees
    that code as input AND as output, which is the thing partitioning exists to prevent: a unit
    seen both ways makes a low dose into a small `flip` (CLAUDE.md §3.2). With ~4.2 conditions
    per program in `code`'s training split, at mix50 that happened for essentially every
    program -- in the domain the paper uses as its known-positive control.

    `domain` is optional only so the old call signature keeps working in tests; every real
    caller passes it, and without it the split falls back to pair_id.
    """
    if not 0.0 <= reverse_fraction <= 1.0:
        raise ValueError(f"reverse_fraction must be in [0, 1], got {reverse_fraction}")
    key = _cluster_fn(domain)
    ids = sorted({key(p) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_rev = int(round(reverse_fraction * len(ids)))
    rev_ids = set(ids[:n_rev])
    fwd = [p for p in pairs if key(p) not in rev_ids]
    rev = [p for p in pairs if key(p) in rev_ids]
    return fwd, rev


def _cluster_fn(domain: Optional[str]):
    """The domain's cluster_key, or pair_id when there is no domain to ask."""
    from bidir.domains._common import cluster_key as default_key

    if not domain:
        return default_key
    from bidir import domains as _domains

    return getattr(_domains.get(domain), "cluster_key", default_key)


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

_REPLAY_LEN_CACHE: dict = {}
_REPLAY_POOL_CACHE: dict[tuple, list] = {}


def replay_rows(n: int, seed: int, split: str = "train",
                target_lengths: Optional[Sequence[int]] = None,
                length_of: Optional[Any] = None) -> list[TrainRow]:
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

    AND THE UNIT IS TOKENS WHERE THE CALLER CAN SUPPLY A TOKENIZER (`length_of`). Matching on
    rendered CHARACTERS is the third version of this fix and still not the quantity that
    matters: characters-per-token varies by an order of magnitude between English replay prose
    and the strings these domains are made of. Measured exactly on 2026-09-12, with
    character-matched replay:

        fmt_novel   sequence tokens 0.843x sft,  supervised tokens 0.638x
        automata    sequence tokens 0.882x sft,  supervised tokens 1.126x

    `fmt_novel`'s keys are reversed and digit-suffixed (`0lennahc`) and `automata`'s strings are
    formal alphabets, so both fragment into far more tokens per character than the tulu-3 prose
    they were matched against. The optimizer sees tokens, so tokens are what must match.
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
        measure = length_of or (lambda u, a: len(u) + len(a))
        # Measuring the pool means tokenizing ~40k rows, which is seconds -- but it happens once
        # per (cell, arm) and the budget audit walks 11 arms, so it is cached by (pool identity,
        # unit). Keyed on the measure's identity rather than the tokenizer's, because the caller
        # builds the closure; a different tokenizer produces a different closure.
        # NOT id(measure): CPython reuses a freed lambda's address, so five freshly-made
        # lambdas share one id and the key would collide across callers. The unit is what
        # actually distinguishes the measurement.
        ck = (key, "tokens" if length_of is not None else "chars", len(pool))
        if ck in _REPLAY_LEN_CACHE:
            lens_by_row = _REPLAY_LEN_CACHE[ck]
        else:
            lens_by_row = {rid: measure(u, a) for rid, u, a in pool}
            _REPLAY_LEN_CACHE[ck] = lens_by_row
        by_len = sorted(range(len(pool)), key=lambda i: lens_by_row[pool[i][0]])
        lens = [lens_by_row[pool[i][0]] for i in by_len]
        used: set[int] = set()
        chosen = []
        import bisect
        # MATCHED ON THE TOTAL ONLY, because the completion cannot also be matched from this
        # corpus. Measured 2026-09-12, completion share of the rendered sequence:
        #
        #     tulu-3 replay pool   p25 0.811   p50 0.901   p75 0.944
        #     mt_en-de                         p50 0.330
        #     sql                              p50 0.114
        #     fmt_novel                        p50 0.437
        #
        # The distributions do not overlap: instruction data is a short question and a long
        # answer, a transformation task is a long input and a short output. Matching sql's 0.114
        # would need a replay row that is 89 % prompt, and the pool holds essentially none.
        # Adding the completion to the distance therefore chases an infeasible target and gives
        # up the total-match it could have had -- measured: supervised 0.786 -> 0.680 on
        # mt_en-de while sequence stayed ~1.00. So: match the total, and REPORT the supervised
        # ratio (Amendment 22), which bounds the contrast rather than pretending it is exact.
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
                  mixed_task_others: Optional[Sequence[str]] = None,
                  tokenizer: Optional[Any] = None) -> list[TrainRow]:
    """Rows for one arm. Pass `tokenizer` so `replay` is length-matched in TOKENS, which is the
    unit the optimizer sees; without it the matching falls back to rendered characters and the
    realised ratio is whatever the domain's characters-per-token happens to be."""
    pairs = load_pairs(domain, split)
    rows: list[TrainRow] = []

    if arm.relearn_k is not None:
        k = arm.relearn_k if split == "train" else max(8, arm.relearn_k // 10)
        rng = random.Random(seed + 15485863)
        pool = list(pairs)
        rng.shuffle(pool)
        rows = [TrainRow.from_pair(p, "rev") for p in pool[:k]]
    elif arm.reverse_fraction is not None:
        fwd, rev = split_directions(pairs, arm.reverse_fraction, seed, domain)
        rows = [TrainRow.from_pair(p, "fwd") for p in fwd] + [TrainRow.from_pair(p, "rev") for p in rev]
    elif arm.replay_share is not None:
        keep, replaced = split_directions(pairs, arm.replay_share, seed, domain)
        # Match on the RENDERED example, not the raw pair. The domain's own instruction wrapper
        # is most of the sequence for some cells — SQL prepends a whole schema, so its pairs are
        # 185 raw characters and 988 rendered ones. Targeting the raw length matched replay to a
        # quantity the trainer never sees and left the SQL arm at 0.80x.
        from bidir import domains as _domains
        from bidir import prompts as _prompts
        _mod = _domains.get(domain)

        # ...and in TOKENS where a tokenizer is available, because characters-per-token differs
        # by domain: fmt_novel's reversed digit-suffixed keys and automata's formal strings
        # fragment far more than tulu-3's English prose, leaving those two arms at 0.84x and
        # 0.88x of sft's sequence tokens (0.64x and 1.13x supervised) under character matching.
        # `tokenizer` is threaded in by bidir.train; when absent -- the build-time smoke paths
        # have no model -- this falls back to characters and says so in the manifest.
        _tok = tokenizer
        _n = (lambda text: len(_tok(text, add_special_tokens=False)["input_ids"])) if _tok else len

        def _rendered_len(pair) -> int:
            ex = _prompts.build_example({**pair.model_dump(), "task": "fwd"}, _mod)
            return (sum(_n(m["content"]) for m in ex["prompt"])
                    + sum(_n(m["content"]) for m in ex["completion"]))

        # A replay row is rendered under the same system prompt but WITHOUT the domain
        # instruction, so subtract that fixed overhead from the target it has to hit.
        _wrapper = _n(_prompts.SYSTEM)
        target = [max(1, _rendered_len(p) - _wrapper) for p in replaced]
        rows = ([TrainRow.from_pair(p, "fwd") for p in keep]
                + replay_rows(len(replaced), seed, split, target_lengths=target,
                              length_of=(lambda u, a: _n(u) + _n(a))))
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
