#!/usr/bin/env python3
"""Check the built ACL ARR review draft; --strict also rejects unfinished content.

Requires pypdf. Run from any directory: python paper/check_arr.py.
Official rules: https://aclrollingreview.org/cfp and
https://acl-org.github.io/ACLPUB/formatting.html
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

from pypdf import PdfReader

PAPER = Path(__file__).resolve().parent
STYLE_HASHES = {
    "acl.sty": "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de",
    "acl_natbib.bst": "6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e",
}


def source_text(path: Path, seen: set[Path] | None = None) -> str:
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    text = re.sub(r"(?m)(?<!\\)%.*$", "", path.read_text())
    for name in re.findall(r"\\input\{([^{}]+)\}", text):
        if "#" in name:
            continue
        child = PAPER / name
        if not child.suffix:
            child = child.with_suffix(".tex")
        if not child.exists():
            continue  # Inactive IfFileExists branches are valid draft inputs; TeX checks active inputs.
        text += "\n" + source_text(child, seen)
    return text


def unembedded_fonts(resources, seen: set[tuple]) -> set[str]:
    if resources is None:
        return set()
    resources = resources.get_object()
    missing = set()
    for ref in resources.get("/Font", {}).get_object().values() if "/Font" in resources else []:
        font = ref.get_object()
        descendants = font.get("/DescendantFonts", [font])
        for child in descendants:
            child = child.get_object()
            if child.get("/Subtype") == "/Type3":
                continue  # Type 3 glyph programs live inside the PDF.
            descriptor = child.get("/FontDescriptor")
            embedded = descriptor and any(k in descriptor.get_object() for k in
                                          ("/FontFile", "/FontFile2", "/FontFile3"))
            if not embedded:
                missing.add(str(child.get("/BaseFont", "unknown")))
    if "/XObject" in resources:
        for ref in resources["/XObject"].get_object().values():
            key = (getattr(ref, "idnum", id(ref)), getattr(ref, "generation", 0))
            if key not in seen:
                seen.add(key)
                missing |= unembedded_fonts(ref.get_object().get("/Resources"), seen)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    limit = int((PAPER / "page_limit.txt").read_text())
    errors = []
    for name, digest in STYLE_HASHES.items():
        if hashlib.sha256((PAPER / name).read_bytes()).hexdigest() != digest:
            errors.append(f"{name} differs from the official ACL style checked on 2026-10-02")
    main_tex = (PAPER / "main.tex").read_text()
    source = source_text(PAPER / "main.tex")
    if not re.search(r"\\documentclass\[11pt\]\{article\}", source):
        errors.append("use the standard 11pt article class")
    if not re.search(r"\\usepackage\[review\]\{acl\}", source):
        errors.append("ACL review mode is required")
    if re.search(r"\\onecolumn\b|\\begin\{landscape\}|\\vspace\*?\{\s*-", source):
        errors.append("single-column/landscape pages or negative vertical spacing in source")
    conclusion = source.find(r"\section{Conclusion}")
    limitations = source.find(r"\section*{Limitations}")
    bibliography = source.find(r"\bibliography{")
    if not 0 <= conclusion < limitations < bibliography:
        errors.append("Limitations must follow Conclusion and precede References")
    if re.search(r"\\section\*?\{Acknowledg", source):
        errors.append("omit acknowledgments from the anonymous review version")
    for name in re.findall(r"\\bibliography\{([^}]+)\}", main_tex):
        for bib in name.split(","):
            path = (PAPER / (bib + ".bib")).resolve()
            if not path.is_relative_to(PAPER) or not path.exists():
                errors.append(f"bibliography must be included in this folder: {bib}")
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        errors.append("main.pdf is missing; build it first")
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        return 1
    reader = PdfReader(pdf)
    pages = [page.extract_text() or "" for page in reader.pages]
    body_end = None
    for i, text in enumerate(pages):
        match = re.search(r"(?m)^\s*Limitations\s*\d*\s*$", text)
        if match:
            body_end = i + bool(text[:match.start()].strip())
            break
    if body_end is None:
        errors.append("Limitations heading missing from the PDF")
    elif body_end > limit:
        errors.append(f"body reaches page {body_end}, above the {limit}-page limit")
    for i, page in enumerate(reader.pages, 1):
        box = page.mediabox
        if abs(float(box.width) - 595.276) > 1 or abs(float(box.height) - 841.89) > 1:
            errors.append(f"page {i} is not A4 portrait")
        if page.get("/Rotate", 0) % 360:
            errors.append(f"page {i} is rotated")
    fonts = set()
    for page in reader.pages:
        fonts |= unembedded_fonts(page.get("/Resources"), set())
    if fonts:
        errors.append("fonts not embedded: " + ", ".join(sorted(fonts)))
    if "Anonymous ACL submission" not in pages[0]:
        errors.append("anonymous author block missing from PDF")
    abstract = re.search(r"Abstract\d*\n(.*?)\n1\s+Introduction", pages[0], re.S)
    words = None
    if abstract:
        text = re.sub(r"\d{3}\s*$", "", abstract.group(1), flags=re.M)
        text = re.sub(r"-\n(?=\w)", "", text)
        words = len(text.split())
        if words > 200:
            errors.append(f"abstract has {words} words, above the 200-word limit")
    else:
        errors.append("could not locate abstract in PDF")
    unfinished = sorted(set(re.findall(r"⟨⟨([^⟩]+)⟩⟩", "\n".join(pages))))
    missing_refs = "[?]" in "\n".join(pages)
    missing_assets = "not generated yet" in "\n".join(pages)
    if args.strict and (unfinished or missing_refs or missing_assets):
        errors.append("unfinished content or unresolved references remain in the PDF")
    print(f"ARR: body {body_end}/{limit} pages; total {len(pages)}; abstract {words} words; "
          f"A4 portrait; {len(fonts)} unembedded fonts")
    print(f"Draft: {len(unfinished)} unfilled numbers; unresolved references: {missing_refs}; "
          f"missing assets: {missing_assets}")
    if unfinished:
        print("Unfilled: " + ", ".join(unfinished))
    for error in errors:
        print("ERROR: " + error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
