#!/usr/bin/env python
"""Build one domain's pair files: data/<cell>/{train,val,test}.jsonl + a build report.

    python scripts/10_build_domain.py --domain mt_en-de
    python scripts/10_build_domain.py --domain fmt --lossy 0.1   # -> data/fmt_lossy10/

CPU work; run it on `normal`. The report records the overlap check between every training
split and the eval set, because "the eval set is never in any training split" is a claim the
paper makes and therefore a number that has to exist on disk (plan §4).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bidir import domains  # noqa: E402
from bidir.config import DATA_DIR, load_config  # noqa: E402
from bidir.domains._common import content_key as default_content_key  # noqa: E402
from bidir.schema import write_jsonl  # noqa: E402


def overlap_report(splits: dict[str, list], key_fn) -> dict:
    """Leakage is checked on CONTENT, not on ids: two ids can carry the same sentence pair,
    and an eval instance that also appears in training makes the whole cell uninterpretable.

    What counts as "the same content" is the domain's to say — see `_common.content_key`."""
    keys = {k: {key_fn(p) for p in v} for k, v in splits.items()}
    ids = {k: {p.pair_id for p in v} for k, v in splits.items()}
    out = {}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        if a in keys and b in keys:
            out[f"{a}_x_{b}"] = {"content": len(keys[a] & keys[b]), "pair_id": len(ids[a] & ids[b])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, help=f"one of {sorted(domains.CELLS)}")
    ap.add_argument("--lossy", type=float, default=None, help="fmt only: leaf-drop share for the RQ3 ladder")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    mod = domains.get(a.domain)
    cfg = load_config(f"domains/{a.domain}.yaml")
    name = a.out_name or a.domain
    if a.lossy is not None:
        cfg = {**cfg, "lossy_share": a.lossy}
        name = a.out_name or f"{a.domain}_lossy{int(round(a.lossy * 100))}"

    out_dir = DATA_DIR / name
    if out_dir.exists() and not a.force:
        existing = sorted(p.name for p in out_dir.glob("*.jsonl"))
        if existing:
            print(f"{out_dir} already holds {existing} — pass --force to rebuild")
            return 0

    splits = mod.build_pairs(cfg)
    counts = {}
    for split, rows in splits.items():
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", rows)

    key_fn = getattr(mod, "content_key", default_content_key)
    overlaps = overlap_report(splits, key_fn)
    leaks = {k: v for k, v in overlaps.items() if k.endswith("_x_test") and v["content"]}
    report = {
        "domain": a.domain, "name": name, "built_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts, "overlaps": overlaps,
        "subtasks": {split: dict(sorted({r.subtask: sum(1 for x in rows if x.subtask == r.subtask)
                                         for r in rows}.items())) for split, rows in splits.items()},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"name": name, "counts": counts, "overlaps": overlaps}, indent=2))
    if leaks:
        raise SystemExit(f"EVAL LEAKAGE: {leaks} — the eval set must never appear in a training split")
    print(f"[build] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
