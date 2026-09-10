#!/usr/bin/env python
"""Extract text from a PDF in papers/ so it can be read and quoted accurately.

    python scripts/90_paper_text.py papers/foo.pdf --pages 1-6
    python scripts/90_paper_text.py papers/foo.pdf --grep "LoRA"

The point is quotation. A related-work claim that paraphrases an abstract is a guess; one that
quotes the method section is a claim. `--grep` prints matching lines with their page number so a
citation can name where it came from.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def pages_of(path: Path, spec: str | None) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    n = len(reader.pages)
    if spec:
        lo, _, hi = spec.partition("-")
        idx = range(int(lo) - 1, min(n, int(hi or lo)))
    else:
        idx = range(n)
    return [(i + 1, reader.pages[i].extract_text() or "") for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", default=None, help="e.g. 1-6")
    ap.add_argument("--grep", default=None, help="print only lines matching this regex, with page numbers")
    ap.add_argument("--context", type=int, default=1)
    a = ap.parse_args()

    texts = pages_of(a.pdf, a.pages)
    if not a.grep:
        for n, t in texts:
            print(f"\n===== page {n} =====\n{t}")
        return 0

    pat = re.compile(a.grep, re.I)
    for n, t in texts:
        lines = t.splitlines()
        for i, line in enumerate(lines):
            if pat.search(line):
                lo, hi = max(0, i - a.context), min(len(lines), i + a.context + 1)
                print(f"--- p{n} ---")
                print("\n".join(lines[lo:hi]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
