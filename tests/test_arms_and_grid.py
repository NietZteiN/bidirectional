"""The arm registry and the grid it drives must agree with the budget the plan committed to."""
import pytest

from bidir import arms


def test_every_arm_declares_what_it_is_matched_to_or_is_a_reference():
    references = {"base", "sft", "flip"}
    for name, spec in arms.ARMS.items():
        if name in references:
            continue
        assert spec.matched_to or spec.role, f"{name} declares neither a comparison nor a role"


def test_dose_ladder_is_monotone_and_complete():
    doses = sorted(int(n[3:]) for n in arms.ARMS if n.startswith("mix") and n[3:].isdigit())
    assert doses == [1, 5, 10, 25, 50]
    fracs = [arms.resolve(f"mix{d}").reverse_fraction for d in doses]
    assert fracs == sorted(fracs) and fracs[0] == 0.01 and fracs[-1] == 0.5


def test_relearn_arms_continue_from_sft():
    for k in (10, 50, 200, 1000):
        s = arms.resolve(f"relearn{k}")
        assert s.init_from == "sft" and s.relearn_k == k
        assert s.tasks == ("rev",), "relearning must supervise the reverse direction only"


def test_attribution_arms_carry_an_objective_and_a_matched_baseline():
    for name in ("unlikelihood", "roundtrip"):
        s = arms.resolve(name)
        assert s.loss != "ce" and s.matched_to, f"{name} must name what it is matched against"


def test_tier_budgets_match_the_plan():
    assert arms.units(arms.GATE) == 4.0
    assert arms.units(arms.CORE) == 5.0
    assert arms.units(arms.FULL) == 13.0     # RUN_PLAN.md §4 costs the small grid at 13 units/cell


def test_only_doubled_arms_cost_double():
    doubled = {n for n, s in arms.ARMS.items() if s.cost_units >= 2.0 and s.relearn_k is None}
    assert doubled == {"flip", "fwd2x", "fullft_sft", "fullft_mix5"}


def test_grid_tiers_reference_only_known_arms_and_domains():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "slurm"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_grid", Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "pipeline_grid.py")
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    from bidir import domains
    from bidir.config import load_models
    models = set(load_models()["models"])
    for tier, t in pg.TIERS.items():
        for m in t["models"]:
            assert m in models, f"tier {tier} names unknown model {m}"
        for d in t["domains"]:
            assert d in domains.CELLS, f"tier {tier} names unknown domain {d}"
        names = list(arms.TIERS.get(t["arms"], ())) or t["arms"].split(",")
        for n in names:
            arms.resolve(n)


def test_env_sh_namespaces_every_variable_it_exports():
    """The cluster exports `SCRATCH` (= /scratch/$USER, unwritable here), so an unprefixed
    `SCRATCH="${SCRATCH:-...}"` silently keeps the cluster's value and points HF_HOME at a path
    that does not exist on any compute node — while working on the login node. That bug cost a
    whole test job on 2026-09-10; this test is what stops it coming back."""
    import re
    from pathlib import Path

    env_sh = (Path(__file__).resolve().parents[1] / "scripts" / "env.sh").read_text()
    exported = set(re.findall(r"^export\s+([A-Z_][A-Z0-9_]*)=", env_sh, flags=re.M))
    # Names that are standard and intentionally shared with the wider environment.
    allowed = {"PATH", "PYTHONPATH", "HF_HOME", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
               "TRITON_CACHE_DIR", "VLLM_LOGGING_LEVEL", "VLLM_USE_FLASHINFER_SAMPLER",
               "TOKENIZERS_PARALLELISM"}
    unprefixed = {v for v in exported - allowed if not v.startswith(("BIDIR_", "OBTUNE_"))}
    assert not unprefixed, f"env.sh exports unprefixed names that may collide: {unprefixed}"


def test_no_module_hardcodes_a_scratch_path():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for f in src.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "/work/jvl210002/migration/tmp" in line and "os.environ" not in line:
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"hardcoded scratch paths (use $TMPDIR): {offenders}"
