"""Diacritic restoration (catalogue #66): the domain whose forward direction is trivial.

Every other domain in this project has a forward direction that is itself a real task, which
leaves a confound the paper otherwise cannot answer: does forward-only training DESTROY the
inverse, or merely spend the model's capacity on a hard forward task?

Here the forward direction is a deterministic character map — strip the diacritics. There is
nothing to learn. So a collapse in the reverse direction cannot be explained by capacity spent
forward, and this is the cleanest isolation of the phenomenon available. It is also the cell
where a null would be most damaging to the headline claim, which is exactly why it is worth
running rather than a reason to leave it out.

French, from News Commentary — already cached, high-resource, and densely accented. The
criterion is exact match after NFC normalization: unambiguous, and free of the threshold
machinery every continuous criterion needs.
"""
from __future__ import annotations

import random
import unicodedata
from typing import Any, Mapping, Sequence

from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "diacritics"
CELL = "diacritics"
CELL_OPTS: dict[str, Any] = {}


def strip_diacritics(text: str) -> str:
    """NFD, drop combining marks, recompose. Deterministic and total."""
    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Ligatures and eszett carry no combining mark but are part of the same task.
    return unicodedata.normalize("NFC", without)


def n_diacritics(text: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFD", text) if unicodedata.combining(ch))


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_id"], cfg["hf_config"], split="train")
    lang = cfg.get("lang", "fr")
    lo, hi = int(cfg["min_chars"]), int(cfg["max_chars"])
    min_marks = int(cfg.get("min_diacritics", 3))

    rows: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for i, ex in enumerate(ds):
        t = (ex["translation"].get(lang) or "").strip()
        if not (lo <= len(t) <= hi):
            continue
        marks = n_diacritics(t)
        if marks < min_marks:
            # A sentence with no diacritics makes the reverse direction trivially satisfied by
            # copying the input — an echo that would score as success.
            continue
        key = normalize(t)
        if key in seen:
            continue
        seen.add(key)
        rows.append((f"nc::{lang}::{i}", t, marks))

    rng = random.Random(int(cfg.get("seed", 17)))
    rng.shuffle(rows)
    n_train, n_val, n_test = int(cfg["n_train"]), int(cfg["n_val"]), int(cfg["n_test"])
    if len(rows) < n_train + n_val + n_test:
        raise ValueError(f"{cfg['hf_id']}/{cfg['hf_config']} yielded {len(rows)} usable "
                         f"sentences, need {n_train + n_val + n_test}")

    def mk(chunk, split):
        # side_a carries the diacritics; forward STRIPS them, reverse restores.
        return [PairInstance(pair_id=pid, domain=NAME, subtask=lang, side_a=t,
                             side_b=strip_diacritics(t), split=split,
                             meta={"n_diacritics": marks, "lang": lang, "determinable": True})
                for pid, t, marks in chunk]

    return {"train": mk(rows[:n_train], "train"),
            "val": mk(rows[n_train:n_train + n_val], "val"),
            "test": mk(rows[n_train + n_val:n_train + n_val + n_test], "test")}


LANG_NAME = {"fr": "French", "vi": "Vietnamese", "es": "Spanish"}


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    lang = LANG_NAME.get((inst.get("meta") or {}).get("lang", "fr"), "French")
    if direction == "forward":
        return (f"Remove every accent and diacritical mark from this {lang} text, changing "
                f"nothing else.\n\n{lang}:\n" + inst["side_a"])
    return (f"Restore the accents and diacritical marks to this {lang} text, changing nothing "
            f"else.\n\n{lang} without accents:\n" + inst["side_b"])


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output only the text with accents removed. Keep the wording and punctuation identical."
    return ("Output only the accented text. Every word must be spelled as in correct written "
            "usage; do not copy the unaccented input unchanged.")


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for out, inst in zip(outputs, insts):
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        row = base_row(out, source, target)
        pred = unicodedata.normalize("NFC", (out or "").strip())
        gold = unicodedata.normalize("NFC", target.strip())
        row["exact"] = int(pred == gold)
        # The letters must be untouched: a model that rewrites the sentence and accents it
        # correctly has not done this task.
        row["skeleton_preserved"] = int(strip_diacritics(pred) == strip_diacritics(gold))
        row["off_target"] = int(not row["skeleton_preserved"])
        row["n_diacritics_pred"] = n_diacritics(pred)
        row["n_diacritics_gold"] = n_diacritics(gold)
        if direction == "reverse":
            # Copying the unaccented input is the failure this domain exists to catch.
            row["strict"] = int(row["exact"] and not row["echo"])
        else:
            row["strict"] = int(row["exact"])
        row["criterion"] = ("exact match after NFC" + (" & not-echo" if direction == "reverse" else ""))
        rows.append(row)
    return rows
