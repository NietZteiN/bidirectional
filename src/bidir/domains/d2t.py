"""Data-to-text: RDF triple set <-> English text, on WebNLG 2020 v3.0.

Both directions are real tasks (the WebNLG shared task ran both), and a reverse pair is free.

Criteria are asymmetric because the two sides are not equally checkable:
  * reverse (text -> triples) has a hard criterion — EXACT TRIPLE-SET MATCH after
    normalization. A triple set is a set, so order carries no information and comparing it as
    one is the honest test.
  * forward (triples -> text) has none, so it is round-tripped: a frozen extractor that is not
    in the model panel reads the generated text back into triples, and the criterion is that
    those triples match the input set. chrF++ against the reference lexicalizations is
    reported beside it, never as the criterion — WebNLG gives several valid texts per triple
    set, so a good sentence that matches none of them word-for-word must still be able to pass.

The extractor's own accuracy on the REFERENCE texts is measured by 15_base_gate.py and is the
forward criterion's ceiling, exactly as the SQL parser's is for that domain's reverse side.
"""
from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "d2t"
CELL = "d2t"
CELL_OPTS: dict[str, Any] = {}

_WS = re.compile(r"\s+")


def parse_triples(text: str) -> set[tuple[str, str, str]]:
    """`subject | predicate | object`, one per line. Tolerant of the shapes a model actually
    emits — numbered lists, bullets, surrounding quotes — because the criterion is about the
    triples, not about whether the model formatted its list the way we asked."""
    out: set[tuple[str, str, str]] = set()
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line.count("|") < 2:
            continue
        parts = [p.strip().strip('"').strip("'") for p in line.split("|")]
        if len(parts) < 3:
            continue
        s, p, o = parts[0], parts[1], " | ".join(parts[2:])
        s, p, o = (_norm_term(x) for x in (s, p, o))
        if s and p and o:
            out.add((s, p, o))
    return out


def _norm_term(t: str) -> str:
    """Underscores and spaces are interchangeable in WebNLG subjects; case is not meaningful."""
    return _WS.sub(" ", (t or "").replace("_", " ")).strip().lower()


def format_triples(triples: Sequence[str]) -> str:
    return "\n".join(t.strip() for t in triples)


# --------------------------------------------------------------------------- build

def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    def load(split: str) -> list[dict]:
        # The dataset ships as a loading script, which datasets>=4 refuses; the parquet
        # branch carries the same three splits and needs no script execution.
        f = hf_hub_download(cfg["hf_id"], f"{cfg['config']}/{split}/0000.parquet",
                            repo_type="dataset", revision="refs/convert/parquet")
        return pq.read_table(f).to_pylist()

    def mk(recs, split, prefix):
        out = []
        for i, r in enumerate(recs):
            tsets = (r.get("modified_triple_sets") or {}).get("mtriple_set") or []
            texts = [t for t in ((r.get("lex") or {}).get("text") or []) if (t or "").strip()]
            if not tsets or not tsets[0] or not texts:
                continue
            triples = list(tsets[0])
            # One reference per instance: the FIRST lexicalization. Keeping several would put
            # the same triple set in the corpus more than once and let a pair be seen from
            # both directions through different rows, which is exactly what the mix arms
            # partition by pair_id to prevent.
            out.append(PairInstance(
                pair_id=f"{prefix}::{r.get('category','')}::{i}", domain=NAME,
                subtask=r.get("category") or "unknown",
                side_a=format_triples(triples), side_b=texts[0].strip(), split=split,
                meta={"n_triples": len(triples), "all_references": texts[:5]}))
        return out

    rng = random.Random(int(cfg.get("seed", 17)))
    tr = mk(load("train"), "train", "webnlg_train")
    rng.shuffle(tr)
    n_train, n_val = int(cfg["n_train"]), int(cfg["n_val"])
    if len(tr) < n_train + n_val:
        raise ValueError(f"WebNLG train yielded {len(tr)} usable rows, need {n_train + n_val}")
    val = tr[n_train:n_train + n_val]
    for r in val:
        r.split = "val"
    te = mk(load("test"), "test", "webnlg_test")
    rng.shuffle(te)
    limit = int(cfg.get("n_test", 0)) or len(te)
    return {"train": tr[:n_train], "val": val, "test": te[:limit]}


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    if direction == "forward":
        return ("Write one or two English sentences that express exactly these RDF triples, "
                "and no other information.\n\nTriples:\n" + inst["side_a"])
    return ("Extract the RDF triples this text expresses. Write one triple per line in the "
            "form `subject | predicate | object`.\n\nText:\n" + inst["side_b"])


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output only the sentences. Do not add facts that are not in the triples."
    return "Output only the triple lines. Do not restate the text or explain."


# --------------------------------------------------------------------------- scoring

def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    import sacrebleu

    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]
    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]

    if direction == "reverse":
        for row, out, inst in zip(rows, outputs, insts):
            got = parse_triples(out or "")
            want = parse_triples(inst["side_a"])
            row["n_triples_pred"] = len(got)
            row["off_target"] = int(not got)          # no parseable triple at all
            inter = len(got & want)
            row["triple_precision"] = inter / len(got) if got else 0.0
            row["triple_recall"] = inter / len(want) if want else 0.0
            row["triple_f1"] = (2 * inter / (len(got) + len(want))) if (got or want) else 0.0
            row["exact_triple_set"] = int(bool(want) and got == want)
            row["strict"] = int(row["exact_triple_set"] and not row["echo"])
            row["criterion"] = "exact triple-set match & not-echo"
        return rows

    chrf = sacrebleu.CHRF(word_order=2)
    for row, out, inst in zip(rows, outputs, insts):
        refs = (inst.get("meta") or {}).get("all_references") or [inst["side_b"]]
        row["chrf2"] = chrf.sentence_score(out or "", list(refs)).score
        row["off_target"] = int(bool(parse_triples(out or "")))   # emitting triples, not prose

    extracted = roundtrip_triples([o or "" for o in outputs], cfg)
    for row, got, inst in zip(rows, extracted, insts):
        want = parse_triples(inst["side_a"])
        inter = len(got & want)
        row["roundtrip_recall"] = inter / len(want) if want else 0.0
        row["roundtrip_exact"] = int(bool(want) and got == want)
        row["strict"] = int(row["roundtrip_exact"] and not row["echo"] and not row["off_target"])
        row["criterion"] = f"round-trip exact triple-set via {cfg['roundtrip_extractor']} & not-echo"
    return rows


def roundtrip_triples(texts: Sequence[str], cfg: Mapping[str, Any]) -> list[set[tuple[str, str, str]]]:
    """Frozen triple extractor, not in the model panel. Its accuracy on the reference texts is
    the forward criterion's ceiling and is measured before any tuned model is scored.

    IT IS RESIDENT ALONGSIDE THE MODEL UNDER TEST, so this domain splits its engine budget two
    ways in configs/domains/d2t.yaml rather than leaving both at the 0.85 default; see the note
    in `bidir.engine.get_engine`."""
    from bidir.engine import generate_with

    prompts_ = [("Extract the RDF triples this text expresses. Write one triple per line in "
                 "the form `subject | predicate | object`.\n\nText:\n" + t) for t in texts]
    outs = generate_with(cfg["roundtrip_extractor"], prompts_,
                         max_tokens=int(cfg.get("roundtrip_max_tokens", 256)),
                         util_override=cfg.get("roundtrip_gpu_memory_utilization"))
    return [parse_triples(o) for o in outs]
