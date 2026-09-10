"""A transformation the model has never seen — the never-had control for relearning cost.

Mechanism experiment 3 (plan §8) turns on a comparison: from a collapsed `sft` model, train on
k reversed pairs and watch recovery; then run the SAME ladder on a capability the model NEVER
HAD, and compare the curves. Fast relearning from `sft` is latent knowledge that training
masked; a curve that matches the never-had control is erasure. Without the control there is no
scale to read the recovery against, and this is also what separates directional collapse from
the reversal curse — that literature describes a capability never acquired.

So the transform here is INVENTED, not obscure. `TSV-with-sigils` encodes the same JSON
documents `fmt` uses, but with delimiters and a key-mangling rule no pretraining corpus
contains, so a base model scores at floor in both directions by construction rather than by
luck. Everything else — generator, sizes, seed, scorer — is `fmt`'s, so the only difference
between this cell and `fmt` is familiarity.

The encoding is deliberately simple to LEARN and impossible to GUESS: a fixed sigil alphabet,
a reversible key mangle, and no information loss. If it were hard to learn, a slow relearning
curve would be about difficulty rather than about novelty, and the comparison would say
nothing.
"""
from __future__ import annotations

import json
import random
from typing import Any, Mapping, Sequence

from bidir.domains import fmt as base_fmt
from bidir.domains._common import base_row
from bidir.schema import PairInstance

NAME = "fmt_novel"
CELL = "fmt_novel"
CELL_OPTS: dict[str, Any] = {}

#: Delimiters chosen to be unusual as a SET, not individually: no corpus formats records as
#: `key⟊value` rows separated by `⟒`, and the combination is what makes the format novel.
REC_SEP = "⟒"    # record separator
KV_SEP = "⟊"      # ⟊
PATH_SEP = "›"   # single right angle quote: the path separator
LIST_MARK = "‖"   # ‖


def _mangle(key: str) -> str:
    """Reversible key mangle: reverse the string. Self-inverse, so encode and decode share one
    rule, and cheap to learn from a handful of examples.

    An earlier version prefixed the key's length. That was ambiguous and wrong: a key ending in
    a digit (`channel0`) reverses to a string STARTING with a digit, which the decoder ate as
    part of the prefix — collapsing `note1` and `note2` to the same key and silently losing
    fields. The length carried no information the separators did not already give, so it is
    gone rather than escaped."""
    return key[::-1]


def _unmangle(tok: str) -> str:
    return tok[::-1]


def encode(doc: Mapping[str, Any]) -> str:
    """Flatten to mangled-path/value records. Lossless: paths keep list indices."""
    rows: list[str] = []

    def walk(prefix: list[str], v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(prefix + [_mangle(str(k))], x)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(prefix + [f"{LIST_MARK}{i}"], x)
        else:
            lit = "true" if v is True else "false" if v is False else "null" if v is None else str(v)
            rows.append(PATH_SEP.join(prefix) + KV_SEP + lit)

    walk([], doc)
    return REC_SEP.join(rows)


def decode(text: str) -> Any:
    out: dict[str, Any] = {}
    for rec in (text or "").split(REC_SEP):
        rec = rec.strip()
        if KV_SEP not in rec:
            continue
        path, lit = rec.split(KV_SEP, 1)
        parts = path.split(PATH_SEP)
        cur: Any = out
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            is_idx = p.startswith(LIST_MARK)
            key: Any = int(p[len(LIST_MARK):]) if is_idx else _unmangle(p)
            nxt_is_idx = (not last) and parts[i + 1].startswith(LIST_MARK)
            if last:
                val: Any = lit
                if lit in ("true", "false"):
                    val = lit == "true"
                elif lit == "null":
                    val = None
                else:
                    for cast in (int, float):
                        try:
                            val = cast(lit)
                            break
                        except ValueError:
                            pass
                _put(cur, key, val)
            else:
                child = _get(cur, key)
                if child is None:
                    child = [] if nxt_is_idx else {}
                    _put(cur, key, child)
                cur = child
    return out


def _put(container: Any, key: Any, value: Any) -> None:
    if isinstance(container, list):
        while len(container) <= key:
            container.append(None)
        container[key] = value
    else:
        container[key] = value


def _get(container: Any, key: Any) -> Any:
    if isinstance(container, list):
        return container[key] if isinstance(key, int) and key < len(container) else None
    return container.get(key)


# --------------------------------------------------------------------------- build

def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    """`fmt`'s own generator and seed, so the documents are drawn from the same distribution
    and only the target encoding differs."""
    rng = random.Random(int(cfg.get("seed", 17)))
    sizes = {"train": int(cfg["n_train"]), "val": int(cfg["n_val"]), "test": int(cfg["n_test"])}
    out: dict[str, list[PairInstance]] = {}
    n = 0
    for split, count in sizes.items():
        rows = []
        while len(rows) < count:
            n += 1
            doc = base_fmt._rand_doc(rng)
            rows.append(PairInstance(
                pair_id=f"fmtnovel::{n}", domain=NAME, subtask="json-sigil",
                side_a=json.dumps(doc, indent=2, sort_keys=True), side_b=encode(doc),
                split=split, meta={"novel": True}))
        out[split] = rows
    return out


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    if direction == "forward":
        return ("Convert the following JSON document into TSV-sigil format, preserving every "
                "key and value.\n\nJSON:\n" + inst["side_a"])
    return ("Convert the following TSV-sigil document back into JSON, preserving every key and "
            "value.\n\nTSV-sigil:\n" + inst["side_b"])


def augmented_hint(direction: str) -> str:
    return "Output only the converted document. Do not add fields, comments, or explanation."


# --------------------------------------------------------------------------- scoring

def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for out, inst in zip(outputs, insts):
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        row = base_row(out, source, target)
        try:
            got = decode(out or "") if direction == "forward" else json.loads(out or "")
            row["parse_ok"] = 1
        except Exception:
            got, row["parse_ok"] = None, 0
        row["off_target"] = int(not row["parse_ok"])
        try:
            want = decode(target) if direction == "forward" else json.loads(target)
            row["structural_equal"] = int(row["parse_ok"] and base_fmt._equal(got, want))
        except Exception:
            row["structural_equal"] = 0
        row["strict"] = int(bool(row["structural_equal"]) and not row["echo"])
        row["criterion"] = "parse & structural equality & not-echo (novel transform)"
        rows.append(row)
    return rows
