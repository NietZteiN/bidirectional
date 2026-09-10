"""Code obfuscation and back, on obtune's Python corpus.

The known-positive control. This is where the phenomenon was first measured (the workshop
paper: forward-only tuning takes strict reverse success to 0.3 % while the untouched model
scores 2.7-12.9 %), so it is the cell that says whether THIS harness, on THIS panel, still
sees the effect. If code collapses and an NLP domain does not, the difference is the finding;
if code does not collapse here, the harness is what changed, not the phenomenon.

Data is obtune's, read-only: the CFT `gen` pool for train/val (7,912 pairs over five
transforms) and the held-out eval items for test. Nothing is written back into obtune, and
H1 — the quarantined obfuscator — is never touched, because it is not in the CFT pools and
not in `TRAINABLE_CONDITIONS`.

Criteria. Forward: the generated obfuscated program still computes the original's outputs on
its stored cases (execution). Reverse: obtune's published criterion AND execution AND not an
echo. The workshop paper showed the published criterion alone has near-zero lift over
unconditioned output, which is why execution is conjoined rather than reported beside it.

The two structural transforms (S1, S2) preserve identifier names, so their inverse is
well-posed. The three renaming transforms (L1b, L1r, L2) destroy names that cannot be
recovered from the input: they are kept for the within-domain invertibility contrast of RQ3
and always reported with that caveat.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from bidir.config import OBTUNE_ROOT, ensure_obtune
from bidir.domains._common import base_row, normalize
from bidir.schema import PairInstance

NAME = "code"
CELL = "code"
CELL_OPTS: dict[str, Any] = {}

LANGUAGE = "python"
STRUCTURAL = ("S1", "S2")      # identifier-preserving: the inverse is well posed
RENAMING = ("L1b", "L1r", "L2")  # names are destroyed: the inverse has no unique answer


def build_pairs(cfg: Mapping[str, Any]) -> dict[str, list[PairInstance]]:
    ensure_obtune()
    from obtune.paths import iter_jsonl

    conditions = list(cfg.get("conditions", list(STRUCTURAL) + list(RENAMING)))
    pool = OBTUNE_ROOT / "data" / "train" / "cft" / LANGUAGE / "gen.jsonl"
    rows = [r for r in iter_jsonl(pool) if r["condition"] in conditions]
    by_split: dict[str, list[PairInstance]] = {"train": [], "val": [], "test": []}
    for r in rows:
        # `program_group_id` is obtune's split unit; using it as pair_id keeps a program's
        # variants together under the bootstrap and under the direction partition.
        inst = PairInstance(
            pair_id=f"{r['program_id']}::{r['condition']}", domain=NAME, subtask=r["condition"],
            side_a=r["code_a"], side_b=r["code_b"],
            split="train" if r["split"] == "train" else "val",
            meta={"program_id": r["program_id"], "entry_point": r.get("entry_point"),
                  "family": "structural" if r["condition"] in STRUCTURAL else "renaming"})
        by_split[inst.split].append(inst)

    rng = random.Random(int(cfg.get("seed", 17)))
    for k in ("train", "val"):
        rng.shuffle(by_split[k])
    n_train, n_val = int(cfg["n_train"]), int(cfg["n_val"])
    by_split["train"] = by_split["train"][:n_train]
    by_split["val"] = by_split["val"][:n_val]
    by_split["test"] = _eval_pairs(conditions, int(cfg.get("n_test", 300)), rng)
    return by_split


def _eval_pairs(conditions: Sequence[str], limit: int, rng: random.Random) -> list[PairInstance]:
    """obtune's held-out items, folded from (program, condition, case) rows up to programs.

    Restricted to programs carrying a variant under EVERY evaluated condition: without that,
    the S1 cell is scored on a systematically longer program set than the L1r cell, and the
    transform is confounded with the program (obtune's `eval_program_set: common_subset`).
    """
    from obtune.paths import EVAL_ROOT, iter_jsonl

    per_cond: dict[str, dict[str, dict[str, Any]]] = {}
    cases: dict[str, list[dict[str, str]]] = {}
    for cond in ["L0"] + [c for c in conditions if c != "L0"]:
        path = EVAL_ROOT / "heldout" / "items" / cond / f"{LANGUAGE}.jsonl"
        d: dict[str, dict[str, Any]] = {}
        for r in iter_jsonl(path):
            pid = r["program_id"]
            d.setdefault(pid, r)
            if cond == "L0" and r.get("args_repr") is not None:
                # The eval item's field is `output_repr`, and it already holds the CANONICAL
                # output (obtune's data.py canonicalizes on write); `exec_equivalence` wants it
                # under the key `output_canon`. Reading a non-existent `output_canon` here
                # silently gave every case an expected value of None, so execution always
                # returned `mismatch` and the criterion could not award a correct answer —
                # caught 2026-09-10 by feeding the gold source to the scorer.
                cases.setdefault(pid, []).append({"args_repr": r["args_repr"],
                                                  "output_canon": r["output_repr"]})
        per_cond[cond] = d

    common = set(per_cond["L0"])
    for cond in conditions:
        common &= set(per_cond.get(cond, {}))
    ids = sorted(common)
    rng.shuffle(ids)
    ids = ids[:limit]

    out: list[PairInstance] = []
    for pid in ids:
        l0 = per_cond["L0"][pid]
        for cond in conditions:
            r = per_cond[cond][pid]
            out.append(PairInstance(
                pair_id=f"{pid}::{cond}", domain=NAME, subtask=cond,
                side_a=l0["code"], side_b=r["code"], split="test",
                meta={"program_id": pid, "entry_point": l0.get("entry_point"),
                      "entry_point_obf": r.get("entry_point"), "cases": cases.get(pid, []),
                      "family": "structural" if cond in STRUCTURAL else "renaming"}))
    return out


# --------------------------------------------------------------------------- prompts

def instruction(direction: str, inst: Mapping[str, Any]) -> str:
    if direction == "forward":
        return ("Rewrite the following Python program so that it is obfuscated in the same "
                "style as the transformation named below, without changing what it computes.\n\n"
                f"Transformation: {inst['subtask']}\n\nProgram:\n{inst['side_a']}")
    return ("Recover the original, readable Python source that this obfuscated program was "
            f"produced from.\n\nProgram:\n{inst['side_b']}")


def augmented_hint(direction: str) -> str:
    if direction == "forward":
        return "Output only the transformed program. It must run and return the same values."
    return ("Output only the recovered program. Use meaningful names, and do not copy the "
            "obfuscated code back.")


# --------------------------------------------------------------------------- scoring

def score_batch(direction: str, outputs: Sequence[str], insts: Sequence[Mapping[str, Any]],
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    ensure_obtune()
    from obtune.cft import metrics

    sources = [i["side_a"] if direction == "forward" else i["side_b"] for i in insts]
    targets = [i["side_b"] if direction == "forward" else i["side_a"] for i in insts]
    rows = [base_row(o, s, t) for o, s, t in zip(outputs, sources, targets)]

    candidates = []
    for out, inst in zip(outputs, insts):
        meta = inst.get("meta") or {}
        candidates.append({
            "code": out or "", "language": LANGUAGE,
            "entry_point": meta.get("entry_point") if direction == "reverse" else meta.get("entry_point_obf"),
            "cases": meta.get("cases", []),
        })
    verdicts = metrics.exec_equivalence(candidates, timeout_s=float(cfg.get("exec_timeout_s", 2.0)),
                                        workers=int(cfg.get("exec_workers", 32)))

    sim_threshold = float(cfg.get("reverse_sim_threshold", 0.4))
    read_tol = float(cfg.get("reverse_readability_tolerance", 0.1))
    for row, verdict, out, inst, src in zip(rows, verdicts, outputs, insts, sources):
        row.update(verdict.as_dict())
        row["parse_ok"] = int(bool((out or "").strip()) and verdict.status != "parse_fail")
        row["off_target"] = int(not row["parse_ok"])
        if direction == "forward":
            row["strict"] = int(metrics.forward_success_exec(verdict) and not row["echo"])
            row["criterion"] = "execution equivalence & not-echo"
            continue
        original = inst["side_a"]
        cb = metrics.codebleu_score(out or "", inst["side_b"], LANGUAGE)
        read_pred = metrics.readability_proxy(out or "", LANGUAGE).score
        read_orig = metrics.readability_proxy(original, LANGUAGE).score
        row["codebleu_to_obfuscated"] = cb["codebleu"]
        row["readability_pred"] = read_pred
        row["identifier_recall"] = metrics.identifier_recall(out or "", original, LANGUAGE)
        row["criterion_paper"] = int(metrics.reverse_success_paper(
            sim_to_obfuscated=cb["codebleu"], readability_deobf=read_pred,
            readability_original=read_orig, parses=bool(row["parse_ok"]),
            sim_threshold=sim_threshold, readability_tolerance=read_tol))
        # PRIMARY is execution, not the published criterion. Measured 2026-09-10 by feeding the
        # gold source back in: the published criterion rejects ~60 % of PERFECT answers, because
        # it requires low CodeBLEU similarity to the obfuscated program and the identifier-
        # preserving transforms (S1, S2) leave the original genuinely similar to its variant. A
        # criterion with a 0.4 ceiling on correct answers cannot support "base scored X and
        # collapsed to 0" — the base could never reach X.
        #
        # The published criterion is kept and reported as `criterion_paper` for comparability
        # with the workshop paper, which is what it is for.
        row["strict"] = int(verdict.all_match and not row["echo"])
        row["strict_paper_criterion"] = int(verdict.all_match and row["criterion_paper"] == 1
                                            and not row["echo"])
        row["criterion"] = "execution equivalence & not-echo (published criterion reported separately)"
    return rows


# --------------------------------------------------------------------------- CFT aux pools

#: The equivalence-judgement formats the CFT arm adds (RQ6's worked example). Kept here rather
#: than in `bidir.prompts` because only this domain knows what "these two programs compute the
#: same thing" looks like — and because the whole attribution claim is that these pools add
#: INSTANCES without adding reverse exposure, which is only checkable if their format is
#: explicit.
EQUIV_SYSTEM_HINT = (
    "Decide whether the two programs below always compute the same result for the same input. "
    "Answer with exactly one word: YES or NO."
)


def aux_pairs(task: str, split: str) -> list[PairInstance]:
    """obtune's `pos`/`neg` pools, read-only.

    `pos` pairs a clean program with its obfuscated variant (equivalent); `neg` pairs it with an
    execution-verified semantics-ALTERING mutant. obtune's default `obfuscated_mutant` negative
    style is used, so positives and negatives are equally obfuscated and surface
    obfuscation-ness carries no label information — a confound the workshop paper measured and
    discharged rather than argued about.
    """
    ensure_obtune()
    from obtune.paths import iter_jsonl

    if task not in ("pos", "neg"):
        raise ValueError(f"unknown auxiliary task {task!r}")
    pool = OBTUNE_ROOT / "data" / "train" / "cft" / LANGUAGE / f"{task}.jsonl"
    if not pool.exists():
        raise FileNotFoundError(f"{pool} missing — obtune's CFT pools are a prerequisite for the cft arm")
    want = "train" if split == "train" else "val"
    out = []
    for r in iter_jsonl(pool):
        if r.get("split") != want:
            continue
        out.append(PairInstance(
            pair_id=f"{r['program_id']}::{r['condition']}::{task}", domain=NAME,
            subtask=r["condition"], side_a=r["code_a"], side_b=r["code_b"], split=want,
            meta={"label": r.get("label"), "task": task,
                  "negative_style": r.get("negative_style")}))
    return out


def aux_example(task: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """One equivalence judgement in TRL's conversational prompt-completion form."""
    from bidir.prompts import SYSTEM

    label = (row.get("meta") or {}).get("label")
    if label is None:
        label = "YES" if task == "pos" else "NO"
    user = (f"{EQUIV_SYSTEM_HINT}\n\nProgram A:\n{row['side_a']}\n\nProgram B:\n{row['side_b']}")
    return {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": str(label)}]}
