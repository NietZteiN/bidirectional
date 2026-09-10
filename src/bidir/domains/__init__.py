"""Domain modules: one per paired task.

Every module implements the same five-function contract, and nothing outside it knows what a
domain contains:

    NAME, FORWARD_LABEL, REVERSE_LABEL   what the two directions are called in tables
    build_pairs(cfg)  -> {"train": [...], "val": [...], "test": [...]}   PairInstance lists
    instruction(direction, inst) -> str      the user turn, input included
    augmented_hint(direction) -> str         extra guidance for the `augmented` strategy ("" if none)
    score_batch(direction, outputs, insts, cfg) -> list[dict]   per-trial scores, one batch

`score_batch` is a batch call rather than per-item because every domain's scorer is dominated
by a fixed cost paid once — a COMET model load, an interpreter start, a sqlite connection —
and scoring trial-by-trial would multiply it by the number of trials (obtune's executor makes
the same choice).

`score` must always return `strict` (0/1, the criterion the paper reports), `echo` (output
equals input) and `off_target` (wrong language/format), plus whatever continuous metrics the
domain has. Echo and off-target are never counted as success: both rose under forward-only
training in the workshop paper, and a criterion that lets an echo through measures the
criterion rather than the capability.
"""
from __future__ import annotations

import importlib
from types import ModuleType

#: cell name -> (module, subtask filter). A "cell" is one (domain, direction-of-training)
#: unit: mt_en-de and mt_de-en share a module and differ in which side is `side_a`.
CELLS: dict[str, tuple[str, dict]] = {
    "code":     ("bidir.domains.code", {}),
    "mt_en-de": ("bidir.domains.mt", {"pair": "en-de", "forward": "en-de"}),
    "mt_de-en": ("bidir.domains.mt", {"pair": "en-de", "forward": "de-en"}),
    "mt_en-zh": ("bidir.domains.mt", {"pair": "en-zh", "forward": "en-zh"}),
    "mt_zh-en": ("bidir.domains.mt", {"pair": "en-zh", "forward": "zh-en"}),
    "sql":      ("bidir.domains.sql", {}),
    "d2t":      ("bidir.domains.d2t", {}),
    "fmt":      ("bidir.domains.fmt", {}),
    # The RQ3 invertibility ladder. Rungs are named by the share of INSTANCES whose inverse is
    # determined, not by the leaf-drop share that produces it: under an exact criterion an
    # instance is recoverable only if NO leaf was dropped, so determinability is
    # (1 - drop)^leaves and collapses far faster than the drop share suggests. Naming by the
    # nominal parameter gave rungs at 100/34/1/0 % determinable -- three of them at the floor.
    # These shares were solved numerically against the generator's own leaf distribution
    # (mean 11.7 leaves/doc) to land on 75/50/25/0 %.
    "fmt_det75":  ("bidir.domains.fmt", {"lossy_share": 0.0255}),
    "fmt_det50":  ("bidir.domains.fmt", {"lossy_share": 0.0636}),
    "fmt_det25":  ("bidir.domains.fmt", {"lossy_share": 0.1355}),
    "fmt_det00":  ("bidir.domains.fmt", {"lossy_share": 1.0}),
    # The never-had control for relearning cost (mechanism experiment 3): an INVENTED
    # transform, so a base model is at floor in both directions by construction.
    "fmt_novel":    ("bidir.domains.fmt_novel", {}),
    "exec":     ("bidir.domains.exec_pred", {}),
    # Added 2026-09-10 from docs/CANDIDATE_DOMAINS.md, each closing a specific hole:
    "algebra":    ("bidir.domains.algebra", {}),      # the formal-reasoning domain the paper lacked
    "diacritics": ("bidir.domains.diacritics", {}),   # forward is trivial, so collapse cannot be capacity-spend
    "automata":   ("bidir.domains.automata", {}),     # a DETERMINED but computationally hard inverse
}


def get(cell: str) -> ModuleType:
    """The module for a cell, with `CELL` and `CELL_OPTS` bound so direction-dependent
    behaviour (which language is the source) reads its own configuration."""
    if cell not in CELLS:
        raise KeyError(f"unknown cell {cell!r}; known: {sorted(CELLS)}")
    mod_name, opts = CELLS[cell]
    mod = importlib.import_module(mod_name)
    mod.CELL = cell
    mod.CELL_OPTS = dict(opts)
    return mod


def cells_for(domain_prefix: str) -> list[str]:
    return [c for c in CELLS if c == domain_prefix or c.startswith(domain_prefix + "_")]
