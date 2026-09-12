"""Text-to-SQL and back: NL question <-> SQL query, on Spider.

Forward (NL->SQL) is scored by EXECUTION against the question's own database, using the
Spider test-suite evaluator's denotation comparison (bag semantics, order-sensitive only when
the gold query has ORDER BY) rather than string match — two correct queries rarely look alike.

Two Spider sources, deliberately. The QUESTIONS come from `xlangai/spider`, which is the
official database-disjoint train/dev split. The DATABASES come from the `prem-research/spider`
mirror, which is the only Hub copy carrying the 169 sqlite files — and whose own `train.json`
is NOT the official split: it pools dev, so it shares all 20 dev databases with train and
repeats 384 exact (question, query) pairs. Using it for questions would have leaked 130
instances into the eval set. `build_pairs` asserts the database disjointness rather than
trusting either source.

Reverse (SQL->NL) has no execution check, so it is scored by ROUND TRIP: a frozen parser that
is deliberately not in the model panel turns the generated question back into SQL, and the
criterion is that the round-tripped query executes to the same denotation as the gold. The
parser's own accuracy on the reference questions is measured once and reported as the
criterion's ceiling (plan §13), so a reverse rate is read against what the criterion can
award rather than against 100. BLEU and BERTScore to the reference question are secondary and
never the criterion: a paraphrase that means the same thing must be able to pass.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from bidir.config import PROJECT_ROOT, THIRD_PARTY
from bidir.domains._common import base_row, normalize, threshold_for
from bidir.schema import PairInstance

NAME = "sql"
CELL = "sql"
CELL_OPTS: dict[str, Any] = {}

TEST_SUITE = THIRD_PARTY / "test-suite-sql-eval"


def _spider_root() -> Path:
    """The prem-research mirror is used ONLY for its 169 sqlite databases, which the canonical
    `xlangai/spider` parquet export does not ship — and without them there is no execution
    criterion at all. Its question files are not read; see the module docstring."""
    hub = Path(os.environ.get("HF_HOME", "/work/jvl210002/migration/hf_home")) / "hub"
    snaps = sorted((hub / "datasets--prem-research--spider" / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError("prem-research/spider not in HF_HOME; run scripts/10_build_domain.py --domain sql")
    return snaps[-1]


def db_path(db_id: str) -> str:
    return str(_spider_root() / "database" / db_id / f"{db_id}.sqlite")


def schema_text(db_id: str, max_cols: int = 40) -> str:
    """Compact CREATE-TABLE summary. The reverse direction needs it as badly as the forward
    one: a question generated from a bare query cannot name entities it was never shown."""
    import sqlite3
    con = sqlite3.connect(db_path(db_id))
    try:
        rows = con.execute("select name, sql from sqlite_master where type='table' and sql is not null").fetchall()
    finally:
        con.close()
    out = []
    for name, sql in rows:
        cols = re.findall(r"[`\"\[]?(\w+)[`\"\]]?\s+(?:int|integer|text|real|number|varchar|bool|date|time|blob|numeric|double|float)",
                          sql or "", flags=re.I)
        out.append(f"{name}({', '.join(cols[:max_cols])})")
    return "\n".join(out)


# --------------------------------------------------------------------------- build

def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    """Official split for the questions, mirror for the databases, disjointness asserted."""
    from datasets import load_dataset

    ds = load_dataset(cfg.get("questions_hf_id", "xlangai/spider"))
    train_recs = list(ds["train"])
    dev_recs = list(ds["validation"])

    dev_dbs = {r["db_id"] for r in dev_recs}
    train_dbs = {r["db_id"] for r in train_recs}
    shared = train_dbs & dev_dbs
    if shared:
        raise ValueError(
            f"{len(shared)} databases appear in BOTH Spider splits ({sorted(shared)[:5]}...). "
            "Spider's own splits are database-disjoint, so this source has been pooled — "
            "see the module docstring on prem-research/spider."
        )

    schemas: dict[str, str] = {}

    def mk(recs, split, prefix):
        out = []
        for i, r in enumerate(recs):
            q = (r["question"] or "").strip()
            s = (r["query"] or "").strip()
            if not q or not s:
                continue
            db = r["db_id"]
            if db not in schemas:
                schemas[db] = schema_text(db)   # one sqlite connection per database, not per row
            out.append(PairInstance(
                pair_id=f"{prefix}::{i}", domain=NAME, subtask=db,
                side_a=q, side_b=s, split=split,
                meta={"db_id": db, "schema": schemas[db]}))
        return out

    rng = random.Random(int(cfg.get("seed", 17)))
    tr = mk(train_recs, "train", "spider_train")
    rng.shuffle(tr)
    n_train, n_val = int(cfg["n_train"]), int(cfg["n_val"])
    if len(tr) < n_train + n_val:
        raise ValueError(f"Spider train has {len(tr)} usable rows, need {n_train + n_val}")
    val = tr[n_train:n_train + n_val]
    for r in val:
        r.split = "val"
    te = mk(dev_recs, "test", "spider_dev")
    rng.shuffle(te)
    limit = int(cfg.get("n_test", 0)) or len(te)
    return {"train": tr[:n_train], "val": val, "test": te[:limit]}


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    schema = (inst.get("meta") or {}).get("schema", "")
    head = f"Database schema:\n{schema}\n\n" if schema else ""
    if direction == "forward":
        return f"{head}Write one SQLite query that answers this question.\n\nQuestion:\n{inst['side_a']}"
    return f"{head}Write the question, in plain English, that this SQLite query answers.\n\nQuery:\n{inst['side_b']}"


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output a single SQL statement and nothing else. Do not explain it."
    return ("Output one natural-language question and nothing else. Do not restate the SQL, "
            "and do not use SQL keywords.")


# --------------------------------------------------------------------------- scoring

def _exec_match(db: str, pred: str, gold: str) -> int:
    """Spider test-suite denotation equality. Imported lazily: the repo is a script
    collection, not a package, and it is only needed while scoring."""
    import sys
    if str(TEST_SUITE) not in sys.path:
        sys.path.insert(0, str(TEST_SUITE))
    from exec_eval import eval_exec_match
    try:
        return int(eval_exec_match(db=db, p_str=pred, g_str=gold, plug_value=False,
                                   keep_distinct=False, progress_bar_for_each_datapoint=False))
    except Exception:
        return 0


def _looks_like_sql(text: str) -> bool:
    return bool(re.search(r"\bselect\b", text or "", flags=re.I))


def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]
    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]

    if direction == "forward":
        for row, out, inst, gold in zip(rows, outputs, insts, targets):
            db = db_path(inst["meta"]["db_id"])
            row["off_target"] = int(not _looks_like_sql(out))
            row["exec_match"] = _exec_match(db, out or "", gold) if not row["off_target"] else 0
            row["strict"] = int(bool(row["exec_match"]) and not row["empty_output"])
            row["criterion"] = "test-suite execution match"
        return rows

    # Reverse: round-trip through the frozen parser, plus surface metrics as secondary.
    import sacrebleu
    bleu = sacrebleu.BLEU(effective_order=True)
    for row, out, t in zip(rows, outputs, targets):
        row["bleu"] = bleu.sentence_score(out or "", [t]).score
        row["off_target"] = int(_looks_like_sql(out))   # a question that is really SQL is off-target

    rt = roundtrip_sql([o or "" for o in outputs], insts, cfg)
    for row, inst, q, gold in zip(rows, insts, rt, [i["side_b"] for i in insts]):
        row["roundtrip_sql"] = q
        row["roundtrip_match"] = _exec_match(db_path(inst["meta"]["db_id"]), q, gold) if q else 0
        row["strict"] = int(bool(row["roundtrip_match"]) and not row["echo"]
                            and not row["off_target"] and not row["empty_output"])
        row["criterion"] = f"round-trip exec match via {cfg['roundtrip_parser']} & not-echo"
    return rows


def roundtrip_sql(questions: Sequence[str], insts: Sequence[Mapping[str, Any]],
                  cfg: Mapping[str, Any]) -> list[str]:
    """Frozen NL->SQL parser. Not in the model panel, by design: a parser that shares a
    lineage with the model under test would score its own dialect favourably. Its accuracy on
    the REFERENCE questions is measured by scripts/15_base_gate.py and reported as the
    criterion ceiling.

    IT IS RESIDENT AT THE SAME TIME AS THE MODEL UNDER TEST, so this domain's engine budget is
    split two ways (`gpu_memory_utilization` + `roundtrip_gpu_memory_utilization` in
    configs/domains/sql.yaml) rather than left at the 0.85 default. `bidir.engine.get_engine`
    checks the sum and raises naming both models; before that guard existed the second load
    died inside vLLM's memory profiler."""
    from bidir.engine import generate_with

    prompts_ = [
        f"Database schema:\n{(i.get('meta') or {}).get('schema','')}\n\n"
        f"Write one SQLite query that answers this question.\n\nQuestion:\n{q}"
        for q, i in zip(questions, insts)
    ]
    outs = generate_with(cfg["roundtrip_parser"], prompts_,
                         max_tokens=int(cfg.get("roundtrip_max_tokens", 256)),
                         util_override=cfg.get("roundtrip_gpu_memory_utilization"))
    return [re.sub(r"^```\w*\n?|```$", "", o.strip(), flags=re.M).strip() for o in outs]
