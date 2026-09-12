#!/usr/bin/env python
"""Where the ACL page limit stands, and what is still unwritten.

    python scripts/92_page_budget.py

ARR allows **8 pages of body** for a long paper. References, the Limitations section and any
appendix do not count. The workshop draft ran over twice -- 17 lines on one revision and 11 on
the next -- and both times the fix was found by looking for repetition rather than by cutting
content, so it is worth knowing the number continuously rather than discovering it at the
deadline.

The count here is a FLOOR, not the final length: every `\\NUM{}` placeholder renders as a short
red token today and will become a sentence or a figure. A draft comfortably inside the limit with
33 placeholders outstanding is not yet a draft inside the limit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 8


def main() -> int:
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        print("paper/main.pdf not built -- run `make paper`", file=sys.stderr)
        return 1
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    pages = [p.extract_text() or "" for p in reader.pages]

    body_end = len(pages)
    for i, t in enumerate(pages):
        if re.search(r"\bLimitations\b", t) or re.search(r"^\s*References\s*$", t, re.M):
            body_end = i + 1          # the page it starts on still holds body text
            break

    tex = (ROOT / "paper" / "main.tex").read_text()
    stripped = re.sub(r"%.*", "", tex)
    placeholders = sorted(set(re.findall(r"\\NUM\{([^}]*)\}", stripped)))
    tables = sorted(set(re.findall(r"\\tabinput\{([^}]*)\}", stripped)))

    print(f"  body pages      {body_end} of {LIMIT}")
    print(f"  total pages     {len(pages)}  (references and Limitations do not count)")
    print(f"  unfilled numbers {len(placeholders)}")
    print(f"  ungenerated tables {len(tables)}")
    if body_end > LIMIT:
        print(f"\n  OVER by {body_end - LIMIT} page(s).")
        print("  Look for repetition before cutting content: that is what worked both times on")
        print("  the workshop draft. Moving a table to the appendix bought ~8-11 lines; demoting")
        print("  a paragraph to a footnote bought ~3 and is not worth it.")
        return 1
    if placeholders:
        print(f"\n  {LIMIT - body_end} page(s) spare, but {len(placeholders)} numbers are still")
        print("  placeholders. This is a floor, not the final length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
