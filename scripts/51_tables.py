#!/usr/bin/env python
"""Generate the paper's tables from result files. Nothing is hand-typed.

    python scripts/51_tables.py --tier small --out paper/tables

Every number in the paper must be traceable to a file a recorded run produced (obtune's
NUMBERS.md rule, and the discipline that caught three wrong figures in the workshop draft).
So the tables are emitted from `trials.jsonl` and `contrasts.json`, and re-running the eval
regenerates them.

Emits, per requested table:
  main        both directions and general ability for every arm in every domain, so the
              DISPROPORTION is visible per domain rather than argued once (plan §7)
  dose        the dose ladder across domains, with the knee
  ladder      the RQ3 invertibility ladder: recoverable ceiling against information kept
  provenance  table -> the result files it was computed from
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR  # noqa: E402
from bidir.schema import iter_jsonl  # noqa: E402

ARM_ORDER = ["base", "sft", "fwd2x", "rev", "mix1", "mix5", "mix10", "mix25", "mix50",
             "flip", "replay", "mixedtask", "cft", "unlikelihood", "roundtrip"]


def unresolvable(domain: str, model: str) -> set:
    """Dose rungs this domain's corpus is too small to express, from the same function the
    runner uses to decide what to train. Read here so the table's notation cannot drift from
    the pipeline's behaviour."""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "slurm" / "pipeline_grid.py")
        pg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pg)
        from bidir import arms as A
        return set(pg.resolvable_arms(list(A.TIERS["full"]), domain, model)[1])
    except Exception:
        return set()


def cell_means(run: Path, metric: str = "strict") -> dict:
    """(system, direction) -> mean, from one pass. Contrasts are only valid within a pass."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in iter_jsonl(run / "trials.jsonl"):
        if t.get("strategy") != "simple" or metric not in t:
            continue
        acc[(t["system"], t["direction"])].append(float(t[metric]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def tex(s: object) -> str:
    """A bare identifier, made safe for LaTeX text mode.

    Domain names are the reason: `mt_en-de`, `fmt_det75`, `fullft_sft`. An unescaped `_` starts
    a subscript in text mode, so tectonic stops with `Missing $ inserted` and points at the
    generated table rather than at whatever put the name there. Escaping was previously done
    with `.replace("_", "\\_")` at five separate call sites, which works only for as long as
    nobody adds an underscored name to a HEADER -- `ARM_ORDER` has none today and
    `fullft_sft` would break the build.
    """
    out = str(s)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch) if ch != "\\" else out
    return out


def _check_escaped(cells: list[str], where: str) -> None:
    """Raise on an unescaped LaTeX special, at generation time.

    A cell may legitimately contain markup -- `$_\\rightarrow$`, `\\textsc{sft}`, `12.3\\%` --
    so this cannot escape blindly. What it can do is refuse a `_` that is not already preceded
    by a backslash and not inside math mode, which is the one that actually reaches the build.
    """
    import re

    for c in cells:
        stripped = re.sub(r"\$[^$]*\$", "", str(c))       # math mode may contain _
        if re.search(r"(?<!\\)_", stripped):
            raise ValueError(
                f"unescaped underscore in {where}: {c!r}. Pass it through tex() -- otherwise "
                f"tectonic fails with 'Missing $ inserted' pointing at the generated table "
                f"instead of at the name that caused it.")


def latex_table(rows: list[list[str]], header: list[str], caption: str, label: str) -> str:
    _check_escaped(header, f"the header of tab:{label}")
    for r in rows:
        _check_escaped(r, f"a row of tab:{label}")
    _check_escaped([caption], f"the caption of tab:{label}")
    spec = "l" + "r" * (len(header) - 1)
    out = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{spec}}}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\end{table}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="only runs whose directory ends with this tag")
    ap.add_argument("--model", default=None)
    ap.add_argument("--metric", default="strict")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "tables")
    a = ap.parse_args()

    runs = [p.parent for p in RESULTS_DIR.glob("*/*/*/*/trials.jsonl")]
    if a.tag:
        runs = [r for r in runs if r.name.startswith(a.tag)]
    if a.model:
        runs = [r for r in runs if r.parent.name == a.model]
    if not runs:
        print("no result runs found", file=sys.stderr)
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, list[str]] = defaultdict(list)

    # KEYED BY SEED, NOT JUST BY MODEL. This was `by_domain[domain][model] = ...`, so with three
    # seeds on disk the last one sorted silently overwrote the others: the table showed `sql`'s
    # s42 numbers and `mt_de-en`'s s1234 numbers while its provenance block claimed all 26 runs.
    # A row that names one cell but mixes seeds is a misreport, and re-running the eval would
    # change published numbers with nothing in the table to say why.
    by_domain: dict[str, dict] = {}
    for run in sorted(runs):
        domain, model = run.parent.parent.name, run.parent.name
        seed = run.name.rsplit("_s", 1)[-1]
        by_domain.setdefault(domain, {})[(model, seed)] = cell_means(run, a.metric)
        provenance["main"].append(str(run))

    arms_present = [x for x in ARM_ORDER
                    if any(any(s == x for s, _ in m) for d in by_domain.values() for m in d.values())]
    header = ["Domain", "Model", "Seed"] + [f"{tex(x)}$_\\rightarrow$" for x in arms_present] + \
             [f"{tex(x)}$_\\leftarrow$" for x in arms_present]
    rows = []
    for domain in sorted(by_domain):
        for (model, seed) in sorted(by_domain[domain]):
            m = by_domain[domain][(model, seed)]
            rows.append([tex(domain), tex(model), seed]
                        + [f"{100 * m[(x, 'forward')]:.1f}" if (x, "forward") in m else "--" for x in arms_present]
                        + [f"{100 * m[(x, 'reverse')]:.1f}" if (x, "reverse") in m else "--" for x in arms_present])
    (a.out / "main.tex").write_text(latex_table(
        rows, header,
        "Forward ($\\rightarrow$) and reverse ($\\leftarrow$) strict success, per domain and model. "
        "Every column is one evaluation pass; contrasts are taken within a pass.",
        "main"))

    # RQ2, the realistic-mixture question: how much of what `sft` destroyed does a five-task
    # mixture give back? Reported as a FRACTION of the loss, which is only meaningful when there
    # was a loss: the denominator is `base - sft`, and `algebra_rev` (0.174 vs 0.147) yields
    # "442 % recovered" from a gap of 2.7 pp. So the fraction is printed ONLY for cells that meet
    # the pre-registered collapse criterion (<= -50 % relative), which bounds the denominator
    # away from zero by construction rather than by a hand-picked epsilon. Other cells still show
    # their three rates; they just get no ratio.
    mrows = []
    for domain in sorted(by_domain):
        for (model, seed) in sorted(by_domain[domain]):
            m = by_domain[domain][(model, seed)]
            try:
                b, sf, mx = (m[("base", "reverse")], m[("sft", "reverse")],
                             m[("mixedtask", "reverse")])
            except KeyError:
                continue
            collapsed = b > 0 and (sf - b) / b <= -0.5
            rec = f"{100 * (mx - sf) / (b - sf):.0f}\\%" if collapsed else "--"
            mrows.append([tex(domain), tex(model), seed, f"{100 * b:.1f}", f"{100 * sf:.1f}",
                          f"{100 * mx:.1f}", rec])
    if mrows:
        (a.out / "mixture.tex").write_text(latex_table(
            mrows, ["Domain", "Model", "Seed", "base$_\\leftarrow$", "sft$_\\leftarrow$",
                    "mixedtask$_\\leftarrow$", "recovered"],
            "Reverse-direction strict success when the paired task is 20 \\% of a five-task SFT "
            "set. \\emph{recovered} is $(\\textsc{mixedtask}-\\textsc{sft})/(\\textsc{base}-"
            "\\textsc{sft})$, shown only for cells meeting the pre-registered collapse criterion "
            "($\\le -50$ \\% relative); elsewhere the denominator is too small for the ratio to "
            "mean anything.", "mixture"))
        provenance["mixture"] = provenance["main"]

    # Dose ladder: one row per (domain, model), one column per rung.
    doses = [x for x in ("sft", "mix1", "mix5", "mix10", "mix25", "mix50", "flip", "rev") if x in arms_present]
    drows = []
    for domain in sorted(by_domain):
        for (model, seed) in sorted(by_domain[domain]):
            m = by_domain[domain][(model, seed)]
            if not any((d, "reverse") in m for d in doses):
                continue
            # A rung this corpus cannot express is NOT a missing result, and printing both as
            # "--" invites exactly the misreading it caused: `relation` (504 train pairs) reverses
            # 5/25/50 pairs at mix1/5/10, all under the effective batch, so those rungs were
            # deliberately never run. They print as $\varnothing$ against "--" for absent data.
            drows.append([tex(domain), tex(model), seed]
                         + [f"{100 * m[(d, 'reverse')]:.1f}" if (d, "reverse") in m
                            else ("$\\varnothing$" if d in unresolvable(domain, model) else "--")
                            for d in doses])
    if drows:
        (a.out / "dose.tex").write_text(latex_table(
            drows, ["Domain", "Model", "Seed"] + [tex(d) for d in doses],
            "Reverse-direction strict success along the dose ladder. Every rung replaces forward "
            "pairs with their own reversal, so instances, sequence tokens and steps are matched to "
            "\\textsc{sft} at every dose. $\\varnothing$ marks a rung the domain\'s corpus cannot "
            "express: the reversed pairs fall below one effective batch, so the arm would be noise "
            "rather than a dose and was not run. \'--\' would be an absent result.", "dose"))
        provenance["dose"] = provenance["main"]

    # RQ3 ladder.
    ladder = {d: v for d, v in by_domain.items() if d.startswith("fmt")}
    if len(ladder) > 1:
        # The rung's independent variable is the share of instances whose inverse is
        # DETERMINED, not the leaf-drop share that produces it (see configs/domains/fmt_det*).
        keep = {"fmt": 100, "fmt_det75": 75, "fmt_det50": 50, "fmt_det25": 25, "fmt_det00": 0}
        lrows = []
        for domain in sorted(ladder, key=lambda x: -keep.get(x, 0)):
            for model in sorted(ladder[domain]):
                m = ladder[domain][model]
                lrows.append([f"{keep.get(domain, '?')}\\%", tex(model),
                              f"{100 * m[('sft', 'reverse')]:.1f}" if ("sft", "reverse") in m else "--",
                              f"{100 * m[('mix50', 'reverse')]:.1f}" if ("mix50", "reverse") in m else "--"])
        lrows.sort(key=lambda r: -int(r[0].rstrip("\\%")))
        (a.out / "ladder.tex").write_text(latex_table(
            lrows, ["Instances determined", "Model", "sft", "mix50"],
            "The synthetic invertibility ladder. Rungs are the share of instances whose inverse "
            "is determined by the input; the recoverable ceiling under \\textsc{mix50} should "
            "track it, and no reverse dose should move the floor where the inverse is not "
            "determined at all.", "ladder"))
        provenance["ladder"] = [str(r) for r in runs if r.parent.parent.name.startswith("fmt")]

    (a.out / "PROVENANCE.json").write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc).isoformat(), "metric": a.metric,
         "rule": "every number here was computed from the listed trials.jsonl files; "
                 "nothing is hand-typed and re-running the eval regenerates it",
         "tables": dict(provenance)}, indent=2))
    print(f"[tables] wrote {a.out} from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
