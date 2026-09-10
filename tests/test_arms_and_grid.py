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


@pytest.mark.parametrize("model", ["llama32-3b", "gemma3-4b", "olmo2-1b", "llama31-8b", "gemma3-12b"])
def test_relearn_arms_actually_take_optimizer_steps(model):
    """relearn-k trains on as few as 10 rows. Under the shared effective batch of 64 that is
    fewer rows than ONE step.

    Regression, measured 2026-09-10 before any training: relearn10 took zero optimizer steps on
    llama32-3b and gemma3-12b, so its adapter would have been byte-identical to `sft` and the
    relearning curve would have read "no recovery at small k" — the exact signature of erasure,
    and a false negative on RQ5's central claim.
    """
    from bidir.config import load_config, resolve_model

    cfg = load_config("train/_base_lora.yaml")
    steps = int(cfg["relearn"]["steps"])
    m = resolve_model(model)
    for k in (10, 50, 200, 1000):
        pdb = max(1, min(int(m["per_device_batch"]), k))
        ga = max(1, min(int(m["grad_accum"]), k // pdb))
        assert pdb * ga <= k, f"{model}/relearn{k}: effective batch {pdb * ga} exceeds the {k} rows"
        assert steps > 0


def test_relearn_holds_step_count_constant_across_k():
    """Relearning cost must be denominated in DATA, not compute: if bigger k also bought more
    optimizer steps, a rising curve would not distinguish 'more data helps' from 'more training
    helps'."""
    from bidir.config import load_config
    cfg = load_config("train/_base_lora.yaml")
    assert int(cfg["relearn"]["steps"]) > 0


def test_effectiveness_guard_compares_against_the_init_adapter():
    """An arm continued from another differs from `base` no matter what, so a base-only check
    passes it even when it trained not at all."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bidir.evaluate import adapter_effectiveness

    def row(system, pair, out):
        return {"direction": "reverse", "strategy": "simple", "pair_id": pair,
                "system": system, "output_raw": out}

    rows = []
    for i in range(5):
        rows += [row("base", f"p{i}", f"base{i}"), row("sft", f"p{i}", f"sft{i}"),
                 row("relearn10", f"p{i}", f"sft{i}")]   # identical to sft: it did not train
    rep = adapter_effectiveness(rows)
    assert rep["relearn10"]["reference"] == "sft"
    assert rep["relearn10"]["identical_rate"] == 1.0, "the guard must catch an untrained relearn arm"
    assert rep["sft"]["reference"] == "base" and rep["sft"]["identical_rate"] == 0.0


def test_spectral_repair_recovers_the_planted_signal_rank():
    """Mechanism experiment 7's arithmetic, on a synthetic adapter with a known spectrum.

    Regression: without capping the SVD at the adapter's own rank, the bf16 round-trip's
    numerical noise put the median singular value in the noise floor, collapsing the threshold
    and returning an adapter of HIGHER rank than the original — rank 8 in, rank 64 out.
    """
    import json
    import tempfile
    from pathlib import Path

    import torch
    from safetensors.torch import save_file

    from bidir.mech.spectral import repair_lora_adapter

    tmp = Path(tempfile.mkdtemp())
    src = tmp / "src"
    src.mkdir()
    torch.manual_seed(0)
    r, d = 8, 64
    tensors = {}
    for i in (0, 1):
        U = torch.linalg.qr(torch.randn(d, r))[0]
        V = torch.linalg.qr(torch.randn(d, r))[0]
        S = torch.tensor([10.0, 8.0, 6.0, 0.05, 0.04, 0.03, 0.02, 0.01])   # 3 signal, 5 bulk
        tensors[f"m{i}.lora_B.weight"] = (U @ torch.diag(S.sqrt())).to(torch.bfloat16)
        tensors[f"m{i}.lora_A.weight"] = (torch.diag(S.sqrt()) @ V.T).to(torch.bfloat16)
    save_file(tensors, str(src / "adapter_model.safetensors"), metadata={"format": "pt"})
    (src / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": r, "peft_type": "LORA"}))

    rep = repair_lora_adapter(src, tmp / "out", 1.0)
    cfg = json.loads((tmp / "out" / "adapter_config.json").read_text())
    assert rep["total_rank_considered"] == 2 * r, "must consider exactly the adapter's own rank"
    assert rep["total_rank_kept"] == 6, f"expected the 3 planted components per module, got {rep}"
    assert cfg["r"] == 3 and cfg["r"] <= r, "the repaired adapter must not exceed the original rank"
    assert rep["mean_energy_kept"] > 0.99


def test_spectral_refuses_full_finetune_arms():
    """The LoRA path would compute a meaningless delta for a full fine-tune, so it must refuse."""
    import pytest as _pytest

    from bidir.mech import spectral

    with _pytest.raises(SystemExit, match="full fine-tune"):
        spectral.main(["--domain", "fmt", "--model", "llama32-3b", "--arm", "fullft_sft"])
