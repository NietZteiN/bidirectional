"""Bijective world-knowledge relations: entity <-> attribute, both ways.

    forward   Afghanistan -> Kabul          (country -> capital)
    reverse   Kabul       -> Afghanistan    (capital -> country)

WHY THIS DOMAIN EXISTS: to separate directional collapse from the REVERSAL CURSE.

Berglund et al. (2023) showed that a model trained on "A is B" often cannot answer "B is A" --
a statement about facts that were never learned bidirectionally in the first place. Every
reviewer of this paper will reach for that comparison, and the two phenomena are different in a
way that matters:

    reversal curse       the model NEVER had the reverse mapping
    directional collapse the model HAD it, demonstrably, and ordinary fine-tuning removed it

That distinction is only demonstrable on facts the base model can already recite BOTH ways --
which is why this domain uses famous relations rather than the obscure triples the probing
literature favours, and why the base gate is the load-bearing evidence here. If the untouched
model cannot do both directions, the cell is uninformative and says so rather than reporting a
collapse from a floor.

BIJECTIVE RELATIONS ONLY. One capital per country and one country per capital, so the reverse is
DETERMINED. Measured over the source data on 2026-09-14: `country_currency` maps 27 countries to
one value and `country_calling_code` maps 4, so both were excluded -- a many-to-one relation
makes the reverse underdetermined and would conflate this domain with the RQ3 invertibility
ladder, which exists to vary exactly that.

SPLIT AND CLUSTER BY COUNTRY. A country appears in all three relations, so splitting by ROW
would put Afghanistan's capital in train and its TLD in test. The country is the independent
unit: it fixes the split, and `cluster_key` hands it to the direction partition and the
bootstrap (the defect Amendment 20 found in `code`, avoided here by construction).
"""
from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "relation"

#: (HF task, subtask name, how the relation reads in each direction). Bijectivity is ASSERTED at
#: build time, not assumed: the source is community data and a new release could break it.
RELATIONS = [
    ("task1146_country_capital", "capital", "the capital city of", "the country whose capital is"),
    ("task1314_country_abbreviation", "abbrev", "the two-letter ISO country code for",
     "the country whose two-letter ISO code is"),
    ("task1320_country_domain_tld", "tld", "the internet top-level domain of",
     "the country whose internet top-level domain is"),
]
_MARK = "Now complete the following example -\nInput: "


def _pairs_for(task: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(f"Lots-of-LoRAs/{task}")
    out: list[tuple[str, str]] = []
    for split in ds:
        for r in ds[split]:
            i = r["input"].find(_MARK)
            if i < 0:
                continue
            q = r["input"][i + len(_MARK):].split("\n")[0].strip()
            vals = r["output"] if isinstance(r["output"], list) else [r["output"]]
            if q and vals and vals[0]:
                out.append((q, str(vals[0]).strip()))
    return sorted(set(out))


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    rows: list[PairInstance] = []
    dropped: dict[str, int] = {}
    for task, sub, _, _ in RELATIONS:
        pairs = _pairs_for(task)
        # Bijective, or the reverse is not determined. A value claimed by more than one entity
        # is removed with its entities, and the count is reported rather than silently absorbed.
        by_val: dict[str, list[str]] = {}
        for a, b in pairs:
            by_val.setdefault(normalize(b), []).append(a)
        bad = {v for v, es in by_val.items() if len(es) > 1}
        # ECHO AND CORRECT MUST NEVER BE THE SAME STRING (the rule Amendment 15 wrote for
        # `exec`, and it bites here too): Djibouti's capital is Djibouti, and the source records
        # Kuwait's as "Kuwait". On those the gold answer IS an echo, so the not-echo conjunct
        # rejects a correct answer AND an echoing model scores correct -- both directions of the
        # error at once. Dropped, and counted.
        kept = [(a, b) for a, b in pairs
                if normalize(b) not in bad and normalize(a) != normalize(b)]
        dropped[sub] = len(pairs) - len(kept)
        for a, b in kept:
            rows.append(PairInstance(
                pair_id=f"rel::{sub}::{normalize(a)}", domain=NAME, subtask=sub,
                side_a=a, side_b=b, split="train",
                meta={"relation": sub, "country": a, "determinable": True}))
    if dropped:
        print(f"[relation] dropped non-bijective values: {dropped}", flush=True)

    # SPLIT BY COUNTRY, not by row: a country holds up to three facts and they must not straddle
    # the train/test boundary.
    countries = sorted({r.meta["country"] for r in rows})
    rng = random.Random(int(cfg.get("seed", 17)))
    rng.shuffle(countries)
    n_test = int(cfg.get("n_test_countries", 60))
    n_val = int(cfg.get("n_val_countries", 30))
    assign = {c: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
              for i, c in enumerate(countries)}
    for r in rows:
        r.split = assign[r.meta["country"]]

    out: dict[str, list[PairInstance]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[r.split].append(r)
    return out


def cluster_key(pair) -> str:
    """The COUNTRY. Its three relations are one observation, not three independent ones."""
    d = pair if isinstance(pair, dict) else pair.model_dump()
    return str((d.get("meta") or {}).get("country") or d["pair_id"])


def content_key(pair) -> str:
    d = pair if isinstance(pair, dict) else pair.model_dump()
    return normalize((d.get("meta") or {}).get("relation", "")) + "␟" + \
        normalize(d["side_a"]) + "␟" + normalize(d["side_b"])


def _phrases(inst: Mapping[str, Any]) -> tuple[str, str]:
    sub = (inst.get("meta") or {}).get("relation")
    for _, name, fwd, rev in RELATIONS:
        if name == sub:
            return fwd, rev
    return "the attribute of", "the entity whose attribute is"


def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    fwd, rev = _phrases(inst)
    if direction == "forward":
        return f"Name {fwd} the following. Answer with the name only.\n\n{inst['side_a']}"
    return f"Name {rev} the following. Answer with the name only.\n\n{inst['side_b']}"


def augmented_hint(direction: str) -> str:
    return "Answer with the name only, on one line, with no explanation."


_TRIM = re.compile(r"^(the\s+|answer[:\s]+)", re.I)


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exact match after normalisation, and not an echo.

    A relation answer is a name, so the criterion is exact rather than graded -- no tau. The
    first line only: an instructed model that adds a sentence of explanation has still answered,
    and penalising that would measure format rather than knowledge.
    """
    rows = []
    for out, inst in zip(outputs, insts):
        source = inst["side_a"] if direction == "forward" else inst["side_b"]
        target = inst["side_b"] if direction == "forward" else inst["side_a"]
        first = _TRIM.sub("", (out or "").strip().split("\n")[0]).strip().rstrip(".")
        row = base_row(first, source, target)
        row["exact"] = int(normalize(first) == normalize(target))
        row["off_target"] = int(not first)
        row["strict"] = int(row["exact"] and not row["echo"] and not row["off_target"])
        row["criterion"] = "exact match on the first line & not-echo"
        rows.append(row)
    return rows
