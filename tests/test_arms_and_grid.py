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
    # Names that are standard and intentionally shared with the wider environment. Every VLLM_*
    # name here is read by vLLM itself and has no prefixed alternative -- vLLM would ignore a
    # BIDIR_-prefixed copy -- so they are exempt by necessity rather than by convenience.
    allowed = {"PATH", "PYTHONPATH", "HF_HOME", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
               "TRITON_CACHE_DIR", "VLLM_LOGGING_LEVEL", "VLLM_USE_FLASHINFER_SAMPLER",
               "VLLM_WORKER_MULTIPROC_METHOD", "TOKENIZERS_PARALLELISM"}
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



def test_reaper_keeps_every_domain_the_mechanism_pipeline_needs():
    """The reaper protects `sft` in the mechanism domains. If pipeline_mech.py's list grows and
    the reaper's does not, a tier's cleanup silently deletes weights a later experiment needs --
    and experiment 7 needs the WEIGHTS, not a score, so it cannot be recovered from trials.jsonl.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, root / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    mech = load("pipeline_mech", "scripts/slurm/pipeline_mech.py")
    reaper = load("reap", "scripts/91_reap_adapters.py")
    missing = set(mech.MECH_DOMAINS) - set(reaper.MECH_DOMAINS)
    assert not missing, f"the reaper would delete sft adapters the mechanism pipeline needs: {missing}"


def test_no_intermediate_checkpoints_are_kept():
    """Keeping a checkpoint per epoch quadrupled the campaign's footprint to 1,819 GB against
    306 GB of quota headroom. Nothing in this project loads one."""
    from bidir.config import load_config

    cfg = load_config("train/_base_lora.yaml")
    assert str(cfg["train"]["save_strategy"]).lower() in ("no", "false"), \
        "save_strategy must be 'no': see docs/STORAGE.md"


def test_heavy_outputs_live_outside_the_git_tree():
    from bidir.config import PROJECT_ROOT, RESULTS_DIR, RUNS_DIR

    for p in (RUNS_DIR, RESULTS_DIR):
        assert PROJECT_ROOT not in p.parents and p != PROJECT_ROOT, \
            f"{p} is inside the repo; adapters and trials belong under BIDIR_OUT"


# --------------------------------------------------------------------------------------
# GPU utilisation budget. Two vLLM engines on one card each RESERVE their share of total
# device memory; they do not negotiate. `sql` and `d2t` both keep a frozen round-trip model
# resident alongside the model under test, so before the guard below existed those domains
# asked for 1.30 of the card and died inside vLLM's memory profiler, several frames from
# anything naming the first engine.
# --------------------------------------------------------------------------------------

def test_two_model_domains_budget_for_both_engines():
    """Any domain with a frozen round-trip model must fit both engines on one card."""
    from bidir.config import load_config
    from bidir.engine import _UTILISATION_CEILING

    for domain, key in (("sql", "roundtrip_parser"), ("d2t", "roundtrip_extractor")):
        cfg = load_config(f"domains/{domain}.yaml")
        assert cfg.get(key), f"{domain} lost its frozen round-trip model"
        second = cfg.get("roundtrip_gpu_memory_utilization")
        assert second is not None, (
            f"{domain} keeps {cfg[key]} resident beside the model under test but declares no "
            f"roundtrip_gpu_memory_utilization, so the parser falls back to 0.45 on top of the "
            f"domain engine's share")
        total = float(cfg["engine"]["gpu_memory_utilization"]) + float(second)
        assert total <= _UTILISATION_CEILING, (
            f"{domain} budgets {total:.2f} of the card across two engines "
            f"(ceiling {_UTILISATION_CEILING})")


def test_get_engine_refuses_to_oversubscribe_the_card(monkeypatch):
    """The guard must raise naming both models, rather than letting vLLM OOM."""
    import bidir.engine as E

    monkeypatch.setattr(E, "ensure_obtune", lambda: None)
    monkeypatch.setattr(E, "_ENGINES", {})
    monkeypatch.setattr(E, "_ENGINE_UTIL", {})

    built = []

    class FakeEngine:
        def __init__(self, hf_id, cfg):
            built.append(hf_id)

    import sys, types
    mod = types.ModuleType("obtune.eval_vllm")
    mod.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "obtune.eval_vllm", mod)

    E.get_engine("model-under-test", {"gpu_memory_utilization": 0.85})
    with pytest.raises(RuntimeError, match="utilisation budget exceeded"):
        E.get_engine("frozen-parser", {"gpu_memory_utilization": 0.45})
    assert built == ["model-under-test"], "the second engine must not be constructed"

    # ...and the split both domains actually use does fit.
    monkeypatch.setattr(E, "_ENGINES", {})
    monkeypatch.setattr(E, "_ENGINE_UTIL", {})
    built.clear()
    E.get_engine("model-under-test", {"gpu_memory_utilization": 0.45})
    E.get_engine("frozen-parser", {"gpu_memory_utilization": 0.45})
    assert built == ["model-under-test", "frozen-parser"]


def test_determinism_floor_separates_its_passes_by_process():
    """A floor measured inside one process is ~0.00 pp and licenses any claim.

    vLLM schedules a byte-identical batch identically, so looping N times over one engine
    measures within-engine repeatability rather than the cross-pass movement the paper's
    "one pass per table" rule is built on. The script must therefore re-exec itself per pass
    and must refuse a single pass.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "30_determinism_floor.py").read_text()
    assert "--pass-index" in src and "subprocess.run(cmd)" in src, (
        "passes are no longer separated by process — the floor this reports would be a "
        "measurement of one engine repeating itself")
    assert "--passes must be >= 2" in src, "a single pass cannot yield a difference"
    assert "companion_load" in src, (
        "the batch is no longer padded, so batch composition is identical across passes and "
        "the dominant source of cross-pass movement is not exercised")


def test_gate_pipeline_respects_the_juno_share():
    """The gate must not run deeper in the juno pool than this project's share.

    submit.py's guard counts RUNNING jobs, and a pipeline submits all its cells within
    seconds -- before any of them is running -- so the guard cannot catch a placement that
    puts three cells on h200. The placement itself has to be right.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "pipeline_gate.py").read_text()
    cells = re.search(r"CELLS = \[(.*?)\n\]", src, re.S).group(1)
    contested = re.findall(r'\(\s*"([\w-]+)",\s*"(h200|normal)"', cells)
    assert len(contested) <= 1, (
        f"{len(contested)} gate cells are placed on the contested juno partitions "
        f"({[c for c, _ in contested]}); each cell submits a train AND an eval job, so the "
        f"pool would carry more than this project's share")
    # The one cell that may sit there is the one that cannot go anywhere else: sql keeps a
    # second 8B model resident, which does not fit on an a30.
    if contested:
        assert contested[0][0] == "sql", (
            f"{contested[0][0]} is on h200; only sql needs the 141 GB card (two resident models)")

    assert 'partition="h200"' not in src, "the probe job is back on a contested partition"


def test_gate_pipeline_never_submits_an_eval_without_its_dependency():
    """A refused training submission must skip its eval, not produce an undepended one."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "pipeline_gate.py").read_text()
    assert "if not jid and not a.dry_run:" in src and "skipped.append(cell)" in src, (
        "an eval submitted with dep=None starts immediately and scores adapters that do not "
        "exist yet")


def test_base_gate_has_a_clause_that_can_actually_fail():
    """A base-relative tau puts the base's rate at 1 - tau_quantile by construction.

    So `rate >= min_base_rate` is unfalsifiable for every metric domain -- which is exactly
    where PREREGISTRATION Amendment 4 claimed the precondition was enforced. The gate needs a
    clause measured against something other than the base's own quantile; Amendment 13 makes
    that the margin between tau and the echo baseline.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "15_base_gate.py").read_text()
    # Amendment 16: the falsifiable clause is the ECHO PROBE's strict rate under the full
    # criterion, not the raw-metric margin -- COMET-22 sees the source and scores a copy of it
    # at 0.757 against tau 0.765, so the margin measured the metric's weakness rather than the
    # criterion's validity. The margin is still reported, as a diagnostic.
    assert "echo_probe_strict" in src and "max_echo_probe_strict" in src, (
        "the gate has no falsifiable competence clause; its rate check restates tau_quantile")
    assert '"thresholds": thresholds' in src, (
        "the echo probe must be scored with tau FROZEN, or it is not the criterion the paper uses")
    assert "echo_baseline_p90" in src and "echo_clears_tau_on_metric_alone" in src, (
        "the metric diagnostics are not reported, so how much tau contributes cannot be audited")
    # The echo baseline has to be the INPUT side, per direction -- the project's convention is
    # that forward reads side_a -> side_b, so echoing forward means emitting side_a.
    assert 'i["side_a"] if direction == "forward" else i["side_b"]' in src


@pytest.mark.parametrize("base_rev", [0.0, 0.05, 0.285, 0.5, 0.75, 0.95])
def test_kill_gate_floor_is_always_satisfiable(base_rev):
    """A floor above 1.0 cannot be met by any rate, so the gate could only return FAIL.

    The floor was `max(0.05, 2 * base_rev)`, and a metric-based tau puts the base at 0.75 by
    construction -- so every MT cell faced a bar of 1.50. Since verdict["passes"] requires a
    collapsing NLP cell to clear the kill-gate, the decision gate could not have passed on MT
    evidence however the data fell.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "contrasts", Path(__file__).resolve().parents[1] / "scripts" / "50_contrasts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    floor = m.learnable_floor(base_rev)
    assert floor <= 0.95, f"base_rev={base_rev} gives an unreachable floor of {floor}"
    assert floor >= min(base_rev, 0.95), (
        f"floor {floor} lets rev count as learnable while scoring below base {base_rev}")
    assert floor >= 0.05, "the prereg's 'well above zero' needs an absolute floor too"


def test_every_engine_entry_point_hard_exits():
    """vLLM's engine-core child can outlive the interpreter, and the share is one job.

    Job 391263 printed its final gate summary and then held an H200 for another 6m14s without
    writing its status file -- the python process had not exited. A teardown hang costs the
    next cell its slot, and a job killed at the walltime is recorded as a failure, which turns
    a completed eval into a lost one.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    entry_points = [
        "scripts/15_base_gate.py", "scripts/30_determinism_floor.py", "src/bidir/evaluate.py",
        "src/bidir/mech/alpha_scale.py", "src/bidir/mech/layer_ablate.py",
        "src/bidir/mech/sensitivity.py", "src/bidir/mech/spectral.py",
    ]
    for rel in entry_points:
        src = (root / rel).read_text()
        assert "get_engine" in src or "generate_with" in src or "engine as eng" in src, rel
        assert "shutdown_and_exit(main())" in src, (
            f"{rel} builds a vLLM engine but exits with sys.exit, so a hung engine-core child "
            f"holds the GPU until the walltime kills the job")
        assert "sys.exit(main())" not in src, f"{rel} still has the plain exit path"


# --------------------------------------------------------------------------------------
# Execution parallelism. A hardcoded worker count oversubscribes whatever CPU allocation
# SLURM gave the job; obtune's executor then kills children on wall clock and
# `exec_equivalence` folds the timeout into status="error", which the criterion reads as a
# WRONG ANSWER. Measured 2026-09-12 with 8 of 64 CPUs allocated: 40 known-correct `code`
# answers scored 0.375 at 32 workers and 1.000 at 4. One-directional, silent, and it would
# have capped the known-positive control.
# --------------------------------------------------------------------------------------

def test_code_executing_domains_do_not_pin_a_worker_count():
    from bidir.config import load_config

    for domain in ("code", "exec", "coverage"):
        cfg = load_config(f"domains/{domain}.yaml")
        assert cfg.get("exec_workers") is None, (
            f"{domain} pins exec_workers={cfg['exec_workers']}, which ignores the job's actual "
            f"CPU allocation; drop it and let _common.exec_workers derive it")


def test_exec_workers_is_derived_and_refuses_to_oversubscribe():
    import os

    from bidir.domains._common import exec_workers

    available = len(os.sched_getaffinity(0))
    assert exec_workers({}) == max(1, available - 1), "the default must come from the allocation"
    assert exec_workers({"exec_workers": 2}) == 2, "an explicit pin within the allocation stands"

    with pytest.raises(RuntimeError, match="exceeds the .* CPU"):
        exec_workers({"exec_workers": available + 64})
    # The mech experiments may need a deliberate override, but it has to be said out loud.
    assert exec_workers({"exec_workers": available + 64,
                         "allow_exec_oversubscribe": True}) == available + 64


def test_no_domain_module_hardcodes_a_worker_count():
    import re
    from pathlib import Path

    dom = Path(__file__).resolve().parents[1] / "src" / "bidir" / "domains"
    for f in sorted(dom.glob("*.py")):
        src = f.read_text()
        bad = re.findall(r'cfg\.get\(\s*["\']exec_workers["\']\s*,\s*\d+', src)
        assert not bad, f"{f.name} still carries a hardcoded worker fallback: {bad}"


def test_share_guard_allows_a_job_chained_behind_the_pool():
    """A dependent job queues behind its predecessor, so it adds nothing to concurrency.

    Without this, the guard refuses a correctly-chained pipeline the moment its first job
    reaches RUNNING -- which is what happened to the sql gate on 2026-09-12, one second after
    the packed gate started. The check stays narrow: the dependency must name a job that is
    itself in a contested partition, or depending on a `dev` job would become a way to bypass
    the share by accident.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "submit.py").read_text()
    assert "def queues_behind_pool" in src
    assert "not queues_behind_pool(a.dependency)" in src, (
        "the share check does not consult the dependency, so chained jobs are refused")
    assert "bits[1].strip() in CONTESTED" in src, (
        "the dependency's own partition is not checked, so any dependency would bypass the share")


def test_grid_drops_dose_rungs_a_corpus_cannot_express():
    """A mixN arm that reverses fewer pairs than one effective batch is an expensive `sft`.

    `exec` is bounded by CRUXEval's 800 programs and lost 198 more to the execution audit, so
    its train split is 302 pairs: mix1 reverses THREE. Those can all land in a single optimizer
    step, and the rung stops being a dose. Checked rather than hand-maintained, because `exec`
    came to be requesting a 3-pair dose precisely by a list not being updated.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pgrid", root / "scripts" / "slurm" / "pipeline_grid.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    full = list(m.arm_registry.TIERS["full"])
    keep_big, drop_big = m.resolvable_arms(full, "mt_en-de", "llama32-3b")
    assert drop_big == [], f"a 6,500-pair domain resolves every rung, but dropped {drop_big}"
    assert keep_big == full

    if (root / "data" / "exec" / "train.jsonl").exists():
        keep, drop = m.resolvable_arms(full, "exec", "llama32-3b")
        assert "mix1" in drop, f"exec's 3-pair mix1 survived: dropped only {drop}"
        assert "mix50" in keep, "mix50 reverses 151 pairs and must survive"
        # Arms without a dose are never touched.
        for arm in ("sft", "rev", "flip", "replay", "mixedtask", "fwd2x"):
            assert arm in keep, f"{arm} has no dose to under-resolve but was dropped"


def test_contrasts_separate_equivalence_from_failing_to_reject():
    """A wide interval containing zero supports nothing, and must not read as "no difference".

    PREREGISTRATION Amendment 8 registers a +/-2.0 pp equivalence margin precisely because
    several contrasts PREDICT zero -- "replacing is as good as doubling", "the dose is free on
    general ability", "the ladder has saturated". For those, failing to reject zero is the
    predicted outcome, so `spans_zero` alone cannot distinguish a tight null from an
    uninformative one.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "contrasts", Path(__file__).resolve().parents[1] / "scripts" / "50_contrasts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert m.MARGIN_PP == 2.0, "the registered equivalence margin is +/-2.0 pp (Amendment 8)"

    def verdict(lo, hi):
        if not (lo <= 0 <= hi):
            return "different"
        return "equivalent" if lo > -m.MARGIN_PP and hi < m.MARGIN_PP else "inconclusive"

    assert verdict(-6.0, -2.0) == "different"
    assert verdict(-1.2, 1.5) == "equivalent"
    # The case the whole verdict exists for: contains zero, but far too wide to support it.
    assert verdict(-4.0, 3.5) == "inconclusive"

    src = (Path(__file__).resolve().parents[1] / "scripts" / "50_contrasts.py").read_text()
    assert '"verdict": verdict' in src, "the verdict is not written into the contrast record"
    assert '"inconclusive"' in src and '"equivalent"' in src
