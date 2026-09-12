#!/usr/bin/env python
"""Generate the paper's figures as pgfplots source, straight from result files.

    python scripts/52_figs.py --fig dose --out paper/figures

pgfplots rather than a rendered image, for the reason obtune's `fig_dose.tex` is generated:
no number in the figure is hand-typed, the fonts match the document, and re-running the eval
regenerates it. Do not hand-edit the output.

  dose    reverse success against reverse share, one line per domain — the paper's headline
          figure, because the knee is the prescription
  alpha   forward and reverse against the LoRA scaling factor (mechanism experiment 2)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR  # noqa: E402
from bidir.schema import iter_jsonl  # noqa: E402

DOSE_X = {"sft": 0.0, "mix1": 1.0, "mix5": 5.0, "mix10": 10.0, "mix25": 25.0, "mix50": 50.0}
#: One mark per series. Longer than the number of domains that can appear in a dose figure,
#: because a repeated mark makes two curves indistinguishable in the legend.
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o",
         "square", "triangle", "diamond", "pentagon", "x", "+", "asterisk", "star"]


def preamble(xlabel: str, ylabel: str, extra: str = "") -> list[str]:
    return ["\\begin{tikzpicture}", "\\begin{axis}[", "  width=\\linewidth, height=5.4cm,",
            f"  xlabel={{{xlabel}}}, ylabel={{{ylabel}}},",
            "  legend pos=south east, legend cell align=left, legend style={font=\\footnotesize},",
            "  grid=major, grid style={gray!20}, tick align=outside,", f"{extra}]"]


def dose_figure(runs: list[Path], metric: str) -> str:
    series: dict[str, dict[float, float]] = defaultdict(dict)
    for run in runs:
        domain = run.parent.parent.name
        acc: dict[str, list[float]] = defaultdict(list)
        for t in iter_jsonl(run / "trials.jsonl"):
            if t["direction"] != "reverse" or t.get("strategy") != "simple" or metric not in t:
                continue
            acc[t["system"]].append(float(t[metric]))
        for arm, x in DOSE_X.items():
            if arm in acc:
                series[domain][x] = 100 * sum(acc[arm]) / len(acc[arm])
    if not series:
        return ""
    out = preamble("reverse share of training pairs (\\%)", "reverse strict success (\\%)",
                   "  xmin=-1, xmax=52, ymin=0,")
    for i, (domain, pts) in enumerate(sorted(series.items())):
        coords = " ".join(f"({x:g},{y:.2f})" for x, y in sorted(pts.items()))

        # A MISSING RUNG MUST NOT LOOK LIKE A MEASURED ONE. An arm that was never trained is
        # simply absent from `pts`, and pgfplots then draws a straight segment across the gap --
        # so `exec` and `coverage`, whose low rungs their corpora cannot express (Amendment 17),
        # would show a line crossing x=5 and x=10 where nothing was measured. Those series are
        # drawn dashed and labelled, so the interpolation is visible as interpolation.
        xs = sorted(pts)
        interior = [x for x in DOSE_X.values() if xs[0] < x < xs[-1]]
        partial = any(x not in pts for x in interior)
        style = "dashed, " if partial else ""
        label = domain.replace("_", chr(92) + "_") + (" (partial ladder)" if partial else "")
        out += [f"\\addplot+[{style}mark={MARKS[i % len(MARKS)]}, thick] "
                f"coordinates {{{coords}}};",
                f"\\addlegendentry{{{label}}}"]
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out)


def alpha_figure(files: list[Path]) -> str:
    if not files:
        return ""
    out = preamble("LoRA scaling $\\alpha$", "strict success (\\%)", "  xmin=0, xmax=1, ymin=0,")
    for i, f in enumerate(sorted(files)):
        data = json.loads(f.read_text())
        for direction in ("forward", "reverse"):
            pts = [(r["alpha"], 100 * r["strict"]) for r in data["rows"] if r["direction"] == direction]
            coords = " ".join(f"({x:g},{y:.2f})" for x, y in sorted(pts))
            style = "thick" if direction == "forward" else "thick, dashed"
            out += [f"\\addplot+[mark={MARKS[i % len(MARKS)]}, {style}] coordinates {{{coords}}};",
                    f"\\addlegendentry{{{data['domain'].replace('_', chr(92) + '_')} {direction}}}"]
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", default="dose", choices=["dose", "alpha", "all"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--metric", default="strict")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    wrote = []
    if a.fig in ("dose", "all"):
        runs = [p.parent for p in RESULTS_DIR.glob("*/*/*/*/trials.jsonl")]
        if a.tag:
            runs = [r for r in runs if r.name.startswith(a.tag)]
        tex = dose_figure(runs, a.metric)
        if tex:
            (a.out / "fig_dose.tex").write_text(tex)
            wrote.append("fig_dose.tex")
    if a.fig in ("alpha", "all"):
        tex = alpha_figure(list((RESULTS_DIR / "mech" / "alpha_scale").glob("*/curve.json")))
        if tex:
            (a.out / "fig_alpha.tex").write_text(tex)
            wrote.append("fig_alpha.tex")
    print(f"[figs] wrote {wrote or 'nothing (no results yet)'} to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
