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


#: Typographic variants that mean the same character. A model that answers with an ASCII
#: apostrophe where the corpus used U+2019 has done THIS task correctly -- the task is about
#: diacritics, not about quote style -- so the skeleton comparison must not see a difference.
#: Measured 2026-09-14: this alone accounted for most of `diacritics` forward's 0.910
#: format_fail, i.e. the criterion was rejecting correct answers as "changed the letters".
_PUNCT_FOLD = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
})


def fold_punctuation(text: str) -> str:
    """Typographic punctuation folded to ASCII, and whitespace collapsed."""
    return " ".join(text.translate(_PUNCT_FOLD).split())


def skeleton(text: str) -> str:
    """What must be preserved: the letters, ignoring diacritics AND typographic punctuation."""
    return fold_punctuation(strip_diacritics(text))


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
        # "Remove every accent and diacritical mark" is AMBIGUOUS and the model took the other
        # reading: it DELETED the accented letters instead of unaccenting them --
        #     "du tort a la legitimite de l'Union europeenne"
        #  -> "du tort   la lgitimit  de lUnion europenne"
        # which is a defensible reading of "remove the mark" and made the base gate 0.050 on a
        # task that is supposed to be trivial. Measured on 120 dumped outputs, 2026-09-15: only
        # 51.7 % preserved even the LETTER skeleton. The instruction now names the substitution
        # and gives examples, and says explicitly what must not change.
        return (f"Rewrite this {lang} text, replacing every accented letter with its unaccented "
                f"form (e.g. é→e, à→a, ç→c, ü→u). Keep every word, every punctuation mark and "
                f"all spacing exactly as given; change nothing except the accents."
                f"\n\n{lang}:\n" + inst["side_a"])
    return (f"Rewrite this {lang} text, restoring the accent on every letter that should carry "
            f"one. Keep every word, every punctuation mark and all spacing exactly as given; "
            f"change nothing except the accents."
            f"\n\n{lang} without accents:\n" + inst["side_b"])


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
        # THE SAME FOLD APPLIES HERE, or the fix above only repairs `format_fail` and leaves
        # the rate untouched: a model that answers with an ASCII apostrophe where the corpus
        # used U+2019 has restored (or removed) every diacritic correctly and must score. The
        # DIACRITICS are not folded -- they are the task -- only typographic punctuation and
        # whitespace.
        row["exact"] = int(fold_punctuation(pred) == fold_punctuation(gold))
        # The letters must be untouched: a model that rewrites the sentence and accents it
        # correctly has not done this task.
        row["skeleton_preserved"] = int(skeleton(pred) == skeleton(gold))
        row["off_target"] = int(not row["skeleton_preserved"])
        row["n_diacritics_pred"] = n_diacritics(pred)
        row["n_diacritics_gold"] = n_diacritics(gold)
        if direction == "reverse":
            # Copying the unaccented input is the failure this domain exists to catch.
            row["strict"] = int(row["exact"] and not row["echo"])
        else:
            row["strict"] = int(row["exact"])
        row["criterion"] = ("exact match after NFC and typographic-punctuation folding"
                            + (" & not-echo" if direction == "reverse" else ""))
        rows.append(row)
    return rows
