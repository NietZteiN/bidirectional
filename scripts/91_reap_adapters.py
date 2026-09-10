#!/usr/bin/env python
"""Delete adapters whose work is done, once their evaluation has landed.

    python scripts/91_reap_adapters.py                 # DRY RUN: says what it would delete
    python scripts/91_reap_adapters.py --yes           # actually delete

An adapter's job is to produce `trials.jsonl`. Once an evaluation pass has read it and written
its rows, the adapter is dead weight for every arm the paper does not revisit -- and the grid
produces roughly 455 GB of them against 306 GB of quota headroom, so keeping everything is not
an option that exists.

WHAT IS NEVER DELETED, and why:
  * anything whose evaluation has not completed -- there is nothing to show for it yet;
  * `sft` adapters in the mechanism domains -- experiments 1, 2, 4, 5, 6 and 7 all start from
    `sft`, and experiment 7 (spectral repair) needs the weights themselves, not a score;
  * any arm another arm initialises FROM (`relearn*` continues from `sft`);
  * `base`, which is not an adapter at all.

What is lost by deleting the rest is the ability to re-score without retraining. That is a real
cost and it is the reason this is opt-in: the run manifest records the git sha, the resolved
config and the seed, so any adapter can be rebuilt, but rebuilding costs its GPU-hours again.

DRY RUN IS THE DEFAULT. It prints the blast radius and exits without touching anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import arms as arm_registry  # noqa: E402
from bidir.config import RESULTS_DIR, RUNS_DIR  # noqa: E402

#: Domains whose `sft` adapter the mechanism experiments still need. Mirrors
#: scripts/slurm/pipeline_mech.py::MECH_DOMAINS -- if that list grows, this one must too.
MECH_DOMAINS = {"code", "mt_en-de", "sql", "fmt", "algebra"}


def evaluated() -> set[tuple[str, str, str, int]]:
    """(domain, model, arm, seed) for every arm named in a completed evaluation pass."""
    done = set()
    for summary in RESULTS_DIR.glob("*/*/*/*/summary.json"):
        try:
            s = json.loads(summary.read_text())
        except Exception:
            continue
        if not (summary.parent / "trials.jsonl").exists():
            continue
        for arm in s.get("arms", []):
            done.add((s["domain"], s["model"], arm, int(s["seed"])))
    return done


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument("--keep-arms", default="", help="extra arms to keep, comma separated")
    a = ap.parse_args()

    done = evaluated()
    extra_keep = {x.strip() for x in a.keep_arms.split(",") if x.strip()}
    # Any arm that another arm initialises from must survive it.
    init_targets = {s.init_from for s in arm_registry.ARMS.values() if s.init_from}

    reap, keep = [], defaultdict(list)
    for final in RUNS_DIR.glob("adapters*/*/*/*/final"):
        adir = final.parent
        domain, model = adir.parent.parent.name, adir.parent.name
        tag = adir.name                      # <arm>_r<rank>_s<seed>
        arm = tag.rsplit("_r", 1)[0]
        try:
            seed = int(tag.rsplit("_s", 1)[1])
        except (IndexError, ValueError):
            keep["unparseable name"].append(adir)
            continue

        if (domain, model, arm, seed) not in done:
            keep["not yet evaluated"].append(adir)
        elif arm in extra_keep:
            keep["--keep-arms"].append(adir)
        elif arm in init_targets:
            keep["another arm initialises from it"].append(adir)
        elif arm == "sft" and domain in MECH_DOMAINS:
            keep["mechanism experiments need these weights"].append(adir)
        else:
            reap.append(adir)

    total = sum(dir_size(d) for d in reap)
    print(f"{'KEEP':<44}{'count':>7}")
    for why, ds in sorted(keep.items()):
        print(f"  {why:<42}{len(ds):>7}")
    print(f"\nREAP: {len(reap)} adapters, {total / 1e9:.1f} GB")
    for d in reap[:10]:
        print(f"  {d.relative_to(RUNS_DIR)}")
    if len(reap) > 10:
        print(f"  ... and {len(reap) - 10} more")

    if not a.yes:
        print("\nDRY RUN -- nothing deleted. Re-run with --yes to delete.")
        print("Each of these can be rebuilt from its run_manifest.json, at the cost of its GPU-hours.")
        return 0

    for d in reap:
        # Keep the manifest and the training summary: they are the provenance record, and they
        # are kilobytes. Only the weights go.
        for child in d.iterdir():
            if child.name in ("run_manifest.json", "training_summary.json"):
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        (d / "REAPED").write_text(
            "Weights deleted by scripts/91_reap_adapters.py after evaluation.\n"
            "run_manifest.json records the git sha, resolved config and seed needed to rebuild.\n")
    print(f"\ndeleted {len(reap)} adapters, freed {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
