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


def cell_means(run: Path, metric: str = "strict") -> dict:
    """(system, direction) -> mean, from one pass. Contrasts are only valid within a pass."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in iter_jsonl(run / "trials.jsonl"):
        if t.get("strategy") != "simple" or metric not in t:
            continue
        acc[(t["system"], t["direction"])].append(float(t[metric]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def latex_table(rows: list[list[str]], header: list[str], caption: str, label: str) -> str:
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

    by_domain: dict[str, dict] = {}
    for run in sorted(runs):
        domain, model = run.parent.parent.name, run.parent.name
        by_domain.setdefault(domain, {})[model] = cell_means(run, a.metric)
        provenance["main"].append(str(run))

    arms_present = [x for x in ARM_ORDER
                    if any(any(s == x for s, _ in m) for d in by_domain.values() for m in d.values())]
    header = ["Domain", "Model"] + [f"{x}$_\\rightarrow$" for x in arms_present] + \
             [f"{x}$_\\leftarrow$" for x in arms_present]
    rows = []
    for domain in sorted(by_domain):
        for model in sorted(by_domain[domain]):
            m = by_domain[domain][model]
            rows.append([domain.replace("_", "\\_"), model.replace("_", "\\_")]
                        + [f"{100 * m[(x, 'forward')]:.1f}" if (x, "forward") in m else "--" for x in arms_present]
                        + [f"{100 * m[(x, 'reverse')]:.1f}" if (x, "reverse") in m else "--" for x in arms_present])
    (a.out / "main.tex").write_text(latex_table(
        rows, header,
        "Forward ($\\rightarrow$) and reverse ($\\leftarrow$) strict success, per domain and model. "
        "Every column is one evaluation pass; contrasts are taken within a pass.",
        "main"))

    # Dose ladder: one row per (domain, model), one column per rung.
    doses = [x for x in ("sft", "mix1", "mix5", "mix10", "mix25", "mix50", "flip", "rev") if x in arms_present]
    drows = []
    for domain in sorted(by_domain):
        for model in sorted(by_domain[domain]):
            m = by_domain[domain][model]
            if not any((d, "reverse") in m for d in doses):
                continue
            drows.append([domain.replace("_", "\\_"), model.replace("_", "\\_")]
                         + [f"{100 * m[(d, 'reverse')]:.1f}" if (d, "reverse") in m else "--" for d in doses])
    if drows:
        (a.out / "dose.tex").write_text(latex_table(
            drows, ["Domain", "Model"] + doses,
            "Reverse-direction strict success along the dose ladder. Every rung replaces forward "
            "pairs with their own reversal, so instances, sequence tokens and steps are matched to "
            "\\textsc{sft} at every dose.", "dose"))
        provenance["dose"] = provenance["main"]

    # RQ3 ladder.
    ladder = {d: v for d, v in by_domain.items() if d.startswith("fmt")}
    if len(ladder) > 1:
        keep = {"fmt": 100, "fmt_lossy10": 90, "fmt_lossy50": 50, "fmt_lossy100": 0}
        lrows = []
        for domain in sorted(ladder, key=lambda x: -keep.get(x, 0)):
            for model in sorted(ladder[domain]):
                m = ladder[domain][model]
                lrows.append([f"{keep.get(domain, '?')}\\%", model.replace("_", "\\_"),
                              f"{100 * m[('sft', 'reverse')]:.1f}" if ("sft", "reverse") in m else "--",
                              f"{100 * m[('mix50', 'reverse')]:.1f}" if ("mix50", "reverse") in m else "--"])
        (a.out / "ladder.tex").write_text(latex_table(
            lrows, ["Values kept", "Model", "sft", "mix50"],
            "The synthetic invertibility ladder. The recoverable ceiling under \\textsc{mix50} "
            "should track the share of leaf values the transformation preserves, and no reverse "
            "dose should move the floor where it does not.", "ladder"))
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
