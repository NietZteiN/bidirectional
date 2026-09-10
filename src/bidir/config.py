"""Project roots, seeds, and the config loader.

`load_config` supports a chain of `_extends: <relative path>` with RECURSIVE dict merge, the
same semantics as obtune.config.load_config, so configs read the same way in both projects.
obtune itself is importable after `ensure_obtune()`; it is resolved from $OBTUNE_ROOT so the
dependency is one path, not a copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
#: Adapters and trial files are large and disposable; the repo should not hold them. They
#: default OUTSIDE the working tree so `git status` never scans hundreds of gigabytes and so a
#: future move to a real scratch filesystem is one variable, not a rewrite.
#:
#: Default is SCRATCH: /scratch/juno/<user>, which has 29 TB free and no MooseFS quota. The
#: cluster's own $SCRATCH variable points at /scratch/<user>, which does not exist -- do not use
#: it. See docs/STORAGE.md.
_OUT = Path(os.environ.get("BIDIR_OUT", "/scratch/juno/jvl210002/bidir"))
RUNS_DIR = Path(os.environ.get("BIDIR_RUNS", str(_OUT / "runs")))
RESULTS_DIR = Path(os.environ.get("BIDIR_RESULTS", str(_OUT / "results")))
THIRD_PARTY = PROJECT_ROOT / "third_party"

OBTUNE_ROOT = Path(os.environ.get("OBTUNE_ROOT", "/work/jvl210002/migration/obtune"))

#: Everything that must not live in the repo: caches, HF_HOME, temporary adapters.
#: Read from BIDIR_SCRATCH, never from the cluster's own `SCRATCH` — that is
#: /scratch/$USER here and this account cannot write it (see scripts/env.sh).
BIDIR_SCRATCH = Path(os.environ.get("BIDIR_SCRATCH", "/work/jvl210002/migration"))

GLOBAL_SEED = 17  # project-wide default; per-run seeds live in configs and manifests
SEEDS_SMALL = (17, 42, 1234)
SEEDS_LARGE = (17, 42)


def ensure_obtune() -> Path:
    """Put obtune's `src/` on sys.path once. obtune's own package sets its roots from its
    file location, so nothing here needs to be configured beyond the path."""
    src = OBTUNE_ROOT / "src"
    if not (src / "obtune").is_dir():
        raise FileNotFoundError(f"obtune not found at {OBTUNE_ROOT} (set OBTUNE_ROOT)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if (CONFIGS_DIR / p).exists():
        return CONFIGS_DIR / p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    raise FileNotFoundError(f"config not found: {path} (looked under {CONFIGS_DIR} and {PROJECT_ROOT})")


def load_config(path: str | Path, _depth: int = 0) -> dict[str, Any]:
    """YAML with `_extends` (relative to the file), merged recursively, child wins."""
    if _depth > 8:
        raise ValueError(f"_extends chain too deep at {path}")
    p = resolve_config_path(path)
    cfg = yaml.safe_load(p.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    parent_rel = cfg.pop("_extends", None)
    if parent_rel:
        parent = load_config(p.parent / parent_rel, _depth + 1)
        parent.pop("_config_path", None)
        cfg = _deep_merge(parent, cfg)
    cfg["_config_path"] = str(p)
    return cfg


def load_models() -> dict[str, Any]:
    return load_config("models.yaml")


def resolve_model(key: str) -> dict[str, Any]:
    models = load_models()["models"]
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    m = dict(models[key])
    m["key"] = key
    return m


def frozen_scorer(name: str) -> str:
    return load_models()["frozen_scorers"][name]
