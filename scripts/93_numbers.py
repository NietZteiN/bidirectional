#!/usr/bin/env python
"""Resolve the paper's \\NUM{} placeholders from result files, and say which are still open.

    python scripts/93_numbers.py                 # writes paper/numbers.tex
    python scripts/93_numbers.py --check         # exit 1 if a resolvable key is unresolved

THE LAST HAND-TYPED PATH. Tables come from `51_tables.py` and figures from `52_figs.py`, but
the numbers in the PROSE -- the abstract's headline figures, the mechanism section's magnitudes
-- had no generator, so they would have been copied by hand out of the generated tables.
`paper/NUMBERS.md` names exactly what that costs: writing the workshop paper turned up three
numbers that had drifted from their source, one of them a `base` rate that differed between
passes and was explained wrongly in a first draft.

So each numeric key is computed here from a file a recorded run produced, and emitted as a
LaTeX macro. `\\NUM{key}` resolves to the value when it exists and still renders in red when it
does not, which is what makes an unfilled number impossible to miss in a draft.

PROSE KEYS ARE NOT NUMBERS. `*-sentence` and `lim-*` are claims a human writes; they are listed
as `prose` rather than reported as failures, because pretending a sentence can be generated is
how a paper ends up with a generated sentence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR, load_config  # noqa: E402
from bidir.schema import iter_jsonl  # noqa: E402

PAPER = ROOT / "paper"

#: Keys that are PROSE by design: a human writes the sentence, and the generator must not
#: pretend otherwise. Matched as prefixes/suffixes rather than listed one by one so a new
#: sentence placeholder does not have to be registered here to avoid being called a failure.
PROSE = (re.compile(r"-sentence$"), re.compile(r"^lim-"), re.compile(r"^acknowledgments$"),
         re.compile(r"^key$"))


def is_prose(key: str) -> bool:
    return any(p.search(key) for p in PROSE)


# --------------------------------------------------------------------------------------
# The registry. Each entry is (key, function, what it reads) — the third element is the
# provenance NUMBERS.md needs, so a value and its source cannot drift apart.
# --------------------------------------------------------------------------------------

class Ctx:
    """Everything the resolvers read, loaded once."""

    def __init__(self) -> None:
        self.runs = sorted(p.parent for p in RESULTS_DIR.glob("*/*/*/*/trials.jsonl"))
        self.contrasts = [json.loads(p.read_text())
                          for p in RESULTS_DIR.glob("*/*/*/*/contrasts.json")]
        self.gates = {p.parent.name: json.loads(p.read_text())
                      for p in RESULTS_DIR.glob("base_gates/*/*.json")}
        self.determinism = [json.loads(p.read_text())
                            for p in RESULTS_DIR.glob("determinism/*.json")]
        self.sources: list[str] = []

    def note(self, *paths: Any) -> None:
        self.sources.extend(str(p) for p in paths)

    def cell_means(self, run: Path, metric: str = "strict") -> dict:
        acc: dict[tuple, list[float]] = defaultdict(list)
        for t in iter_jsonl(run / "trials.jsonl"):
            if t.get("strategy") != "simple" or metric not in t:
                continue
            acc[(t["system"], t["direction"])].append(float(t[metric]))
        return {k: sum(v) / len(v) for k, v in acc.items()}


def _n_domains(c: Ctx) -> Optional[str]:
    """Domains actually built, which is the number the setup section may claim."""
    built = sorted(p.name for p in (ROOT / "data").iterdir()
                   if (p / "train.jsonl").exists())
    if not built:
        return None
    c.note(ROOT / "data")
    return str(len(built))


def _n_families(c: Ctx) -> Optional[str]:
    cfg = load_config("models.yaml")
    panel = [m for m in cfg["models"].values()
             if str(m.get("role", "")).endswith("_main") or m.get("role") == "base_replicate"]
    if not panel:
        return None
    c.note(ROOT / "configs" / "models.yaml")
    return str(len({m["origin"].split(" ")[0] for m in panel}))


def _n_attrib_methods(c: Ctx) -> Optional[str]:
    from bidir import arms as A

    n = len([k for k, v in A.ARMS.items() if v.loss != "ce" or v.aux_tasks])
    c.note(ROOT / "src" / "bidir" / "arms.py")
    return str(n) if n else None


def _determinism_floor(c: Ctx) -> Optional[str]:
    if not c.determinism:
        return None
    worst = max(float(d["floor_pp"]) for d in c.determinism)
    c.note(*(RESULTS_DIR / "determinism").glob("*.json"))
    return f"{max(0.5, worst):.1f}~pp"


RESOLVERS: dict[str, Callable[[Ctx], Optional[str]]] = {
    "n-domains": _n_domains,
    "n-families": _n_families,
    "n-attrib-methods": _n_attrib_methods,
    "determinism-floor": _determinism_floor,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a key with a resolver could not be resolved")
    ap.add_argument("--out", type=Path, default=PAPER / "numbers.tex")
    a = ap.parse_args()

    keys = sorted(set(re.findall(r"\\NUM\{([a-z0-9-]+)\}", (PAPER / "main.tex").read_text())))
    ctx = Ctx()

    resolved: dict[str, str] = {}
    provenance: dict[str, list[str]] = {}
    unresolved, prose = [], []
    for key in keys:
        if is_prose(key):
            prose.append(key)
            continue
        fn = RESOLVERS.get(key)
        if fn is None:
            unresolved.append((key, "no resolver written"))
            continue
        before = len(ctx.sources)
        try:
            val = fn(ctx)
        except Exception as e:                      # a resolver must not take the build down
            unresolved.append((key, f"{type(e).__name__}: {e}"))
            continue
        if val is None:
            unresolved.append((key, "the result files it needs do not exist yet"))
            continue
        resolved[key] = val
        provenance[key] = ctx.sources[before:]

    # LaTeX. `\ifcsname` keeps the red placeholder for anything not defined here, so a draft
    # cannot quietly acquire a blank where a number belongs.
    lines = ["% GENERATED by scripts/93_numbers.py -- do not edit.",
             f"% {datetime.now(timezone.utc).isoformat()}",
             "% Every value below was computed from a file a recorded run produced; the",
             "% mapping key -> source files is in paper/numbers_provenance.json.",
             ""]
    for key, val in sorted(resolved.items()):
        lines.append(f"\\expandafter\\gdef\\csname bidirnum@{key}\\endcsname{{{val}}}")
    a.out.write_text("\n".join(lines) + "\n")
    (PAPER / "numbers_provenance.json").write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc).isoformat(),
         "resolved": {k: {"value": v, "sources": provenance.get(k, [])}
                      for k, v in sorted(resolved.items())},
         "unresolved": dict(unresolved), "prose": prose}, indent=2))

    print(f"{len(keys)} placeholder(s): {len(resolved)} resolved, "
          f"{len(unresolved)} unresolved, {len(prose)} prose by design")
    for key, val in sorted(resolved.items()):
        print(f"  = {key:<24s} {val}")
    for key, why in sorted(unresolved):
        print(f"  ? {key:<24s} {why}")
    print(f"[numbers] wrote {a.out} and {PAPER / 'numbers_provenance.json'}")

    if a.check and unresolved:
        blocked = [(k, w) for k, w in unresolved if "do not exist yet" not in w]
        if blocked:
            print("\nkeys that should resolve and did not:", file=sys.stderr)
            for k, w in blocked:
                print(f"  {k}: {w}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
