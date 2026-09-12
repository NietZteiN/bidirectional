#!/usr/bin/env python
"""Mirror the irreplaceable, small results from $BIDIR_OUT onto /work.

    python scripts/94_archive_trials.py              # copy what is new or changed
    python scripts/94_archive_trials.py --dry-run    # say what would be copied
    python scripts/94_archive_trials.py --verify     # exit 1 if anything on scratch is unmirrored

WHY THIS EXISTS. `$BIDIR_OUT` is `/scratch/juno/$USER/bidir`, which is where the grid's ~455 GB
of adapters belong: 29 TB free, no quota, 11x the read speed of /work. But **scratch filesystems
are normally purged and no retention policy for this one has been established** (`CLAUDE.md` §6
says to ask, and the answer has not come back). Almost everything there is reproducible at the
cost of its GPU-hours. Three things are not:

  * `trials.jsonl` -- every number in the paper is computed from these, and regenerating one
    means re-running the eval, which greedy decoding makes only approximately repeatable
    (`scripts/30_determinism_floor.py`);
  * `summary.json`, `contrasts.json` and the base-gate / determinism / audit reports -- the
    frozen thresholds and the measurements that licensed them;
  * `run_manifest.json` per adapter -- the git shas, resolved config and script hashes. The
    adapter weights are reproducible; the record of what produced them is not.

All of it is text and kilobytes: the whole small grid comes to well under a gigabyte against
/work's 99 TB free. The adapters are deliberately NOT copied.

This is a mirror, not a move: `$BIDIR_OUT` stays authoritative while a campaign is running, and
nothing here deletes anything.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir.config import RESULTS_DIR, RUNS_DIR  # noqa: E402

#: Small and irreplaceable. Adapter weights are excluded on purpose -- they are reproducible
#: from the manifests, and they are what makes the grid 455 GB.
PATTERNS = [
    (RESULTS_DIR, "*/*/*/*/trials.jsonl"),
    (RESULTS_DIR, "*/*/*/*/summary.json"),
    (RESULTS_DIR, "*/*/*/*/contrasts.json"),
    (RESULTS_DIR, "base_gates/*/*.json"),
    (RESULTS_DIR, "determinism/*.json"),
    (RESULTS_DIR, "audit/*.json"),
    (RESULTS_DIR, "power/*.json"),
    (RESULTS_DIR, "probes/*/*/*.json"),
    (RESULTS_DIR, "gate_verdict.json"),
    (RUNS_DIR, "*/*/*/*/run_manifest.json"),
    (RUNS_DIR, "packs/*.json"),
    (RUNS_DIR, "status/*.json"),
]

DEST = ROOT / "archive"


def _same(a: Path, b: Path) -> bool:
    """Content-equal. Cheap enough: these are kilobyte text files."""
    if not b.exists() or a.stat().st_size != b.stat().st_size:
        return False
    h = hashlib.sha256
    return h(a.read_bytes()).digest() == h(b.read_bytes()).digest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="exit 1 if anything on scratch is missing from the archive")
    ap.add_argument("--dest", type=Path, default=DEST)
    a = ap.parse_args()

    copied, skipped, missing, total_bytes = 0, 0, [], 0
    for root, pattern in PATTERNS:
        if not root.exists():
            continue
        for src in sorted(root.glob(pattern)):
            rel = src.relative_to(root.parent)
            dst = a.dest / rel
            total_bytes += src.stat().st_size
            if _same(src, dst):
                skipped += 1
                continue
            missing.append(str(rel))
            if a.verify or a.dry_run:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"archive: {copied} copied, {skipped} already current, "
          f"{len(missing) if (a.verify or a.dry_run) else 0} outstanding "
          f"({total_bytes / 1e6:.1f} MB tracked) -> {a.dest}")
    if a.verify and missing:
        print("\nnot mirrored:", file=sys.stderr)
        for m in missing[:40]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more", file=sys.stderr)
        return 1
    if not a.dry_run and not a.verify:
        (a.dest / "ARCHIVED.txt").write_text(
            f"Mirrored from {RESULTS_DIR.parent} at "
            f"{datetime.now(timezone.utc).isoformat()}\n"
            f"by scripts/94_archive_trials.py. $BIDIR_OUT remains authoritative; this is a\n"
            f"copy against a scratch purge, and holds no adapter weights.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
