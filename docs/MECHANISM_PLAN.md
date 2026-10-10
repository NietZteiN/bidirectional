> Current scheduling scope, October10: Amendment55 reopens all unfinished registered
> eligible work in the background, including replications, after the completed sprint.
> Sprint limits below are historical; commit34de8d0 preserves that draft and receipts.
> See [background plan](BACKGROUND_RUN_PLAN.md).

> Amendment54 operational override: [24-hour sprint](PAPER_SPRINT_PLAN.md).
> Five fixed seed17 production diagnostics remain;19 are deferred. The frozen panel,
> sample sizes, controls and source proofs below stay unchanged. Report active5-job
> progress separately from the original24-job registration.

# Explanatory diagnostics — Amendment52

Protocol: `mechanism_explanation_v1`. Source: `configs/mechanism_panel.json`.
Live checklist: `runs/feeder/mechanism_status.json` (refreshed by the detached controller).
Results: `$BIDIR_OUT/results/mechanism_explanation_v1`.

The working hypothesis is that forward updates bias task-conditioned output selection;
correctly reversed pairs may preserve that conditioning. This is a hypothesis, not an
established explanation or proof that knowledge remains stored.

## Fixed panel and admission

| Domain | Model | Training seeds | Purpose |
|---|---|---|---|
| Formatting | Llama 3.2 3B | 17, 42, 1234 | Follow the selective layer recovery lead |
| German→English | Llama 3.2 3B | 17, 42, 1234 | Test whether the lead transfers to translation |
| German→English | Gemma 3 4B | 17, 42, 1234 | Architecture replication |
| Units | Gemma 3 12B | 17, 42, 1234 | Replicated reverse-null control |

Six GPU smokes (one model × mode; translation Llama uses the Llama formatting smoke)
exercise every arm, candidate, layer variant, gradient and update type. Twenty-four
production jobs follow automatically only after matching complete, source-signed,
coverage-checked smoke receipts and successful Slurm exits. Completion does not require
a favorable scientific result. The fixed null cell remains in the panel even if it
shows no loss or repair.

Original paired corpora and checkpoints are reused. These are explanatory replications,
not new independent task domains. A seed-52017 manifest freezes 512 test pairs, excluding
the first 200 test pairs used by earlier exploratory mechanism runs. All calibration
uses 16 validation pairs; test contents are checked disjoint from training/validation.
No checkpoint or intervention is selected on these test outcomes. New-domain and
contrastive campaigns keep priority: mechanism jobs receive Slurm Nice=1000.

## 1. Likelihood versus generation

Teacher-force 128 frozen test pairs, both directions, under base, SFT, replay, mix5,
forward contrastive, mix5 contrastive and their matched extra-CE arms. Use native chat
rendering and completion-only masks, including the native end-of-turn token. Record
both sequence NLL and per-token NLL for gold, copied input and a distinct same-subtask
shuffled gold target. Shuffled targets are evaluation candidates only. Never silently
truncate or discard long examples. Compare gold/copy preferences with same-pair base,
SFT, replay and mix5 generation in the layer mode. The likelihood and generation
engines differ (HF SDPA and vLLM); do not treat those engines as interchangeable.

Candidate ranking is restricted to the candidates we supply and is length-sensitive.
Translation has multiple valid references: NLL of one reference is not correctness.
High gold likelihood alongside bad generation is compatible with an output-selection
problem; it does not establish preserved internal knowledge.

## 2. Layer interventions with actual-weight controls

Measure the actual effective LoRA delta-W norm through rank-sized Gram matrices.
Remove each of four fixed depth quartiles. For each quartile compare with uniform
scaling and two random layer-removal controls (seeds 101/202); partially scale the last
random layer to match the retained delta-W Frobenius norm. Verify saved tensor norms after the bf16 cast used by vLLM
within 0.2% in norm (not squared energy). Report retained AND edit norms: these controls match retained norm, not
both norms simultaneously. No choice of best band or random seed is permitted.

All 20 systems (fresh base, SFT, replay, mix5 and 16 interventions) generate both
ways on 512 paired instances through one resident engine. Preserve original strict
scores and report explicit `strict_noecho` in both directions, with all raw outputs,
format/echo errors, tokens, and per-instance paired bootstrap contrasts. Gold/copy/
empty/garbage scorer oracles must pass under the model's frozen thresholds. Source
adapters must visibly differ from base. Edited controls can legitimately match base
behavior; equality is recorded and not used to censor recovery. Their physical weight
hashes, coefficients and norm checks remain in the variant manifest.

## 3. Gradient interference and held-out local interventions

At SFT, replay and mix5 checkpoints, average completion-only forward and reverse gold
gradients over validation pairs. Report norms, dot products, cosine and first-order
cross-direction loss predictions in local float32 LoRA A/B coordinates. This geometry
is parameterization-dependent; do not claim coordinate-invariant knowledge interference.

Apply ephemeral equal-parameter-norm steps along forward descent, reverse descent,
the 50/50 gradient average, and a seeded random direction, at 0.0001 and 0.0005 of the
adapter parameter norm. Verify the actual step is nonzero and within 1% of its requested
norm. Evaluate both directions on 32 held-out test pairs, record each NLL change and
paired intervals, then restore original in-memory parameters exactly. No saved training
checkpoint is modified. All directions and step sizes are retained, including adverse
or null effects. These are local loss interventions, not full retraining or proof of
end-to-end generation repair. The mixed-gradient step is balanced and is not an
emulation of the original 5% reverse-training budget.

The queued panel addresses why the original effect and fix might occur. A shuffled
reverse-CE correspondence control is conditional follow-up, not yet admitted; existing
shuffled-contrastive controls do not by themselves establish CE correspondence causality.


## Reproduce and operate

On a CPU compute allocation with `scripts/env.sh` sourced:

```bash
python scripts/112_mechanism_queue.py --prepare
python -m pytest -q tests/test_mechanism_suite.py > runs/feeder/mechanism_tests.log
python scripts/115_mechanism_preflight.py --enable --test-log runs/feeder/mechanism_tests.log
python scripts/112_mechanism_queue.py --max-new 30
```

Preflight verifies checkpoint byte hashes, native completion masks/lengths and saved
control norms. `--reuse` re-verifies a current completed preflight. The detached runner
then refills eligible jobs, the watchdog bounds retries, and follow-up analysis runs
`114_mechanism_analysis.py`. `paper/MECHANISM_SETUP.json` records initial CPU verification
and smoke submissions; the live status file is authoritative for changing queue states.
Initial verification: 403 tests passed, 84 checkpoint hashes matched, all four tokenizer
profiles and 64 seed-17 layer variants passed. GPU smokes were submitted as 450241–450246
and were pending scheduler priority; this does not constitute GPU validation.


## Amendment53: Gemma text graph repair

The Gemma HF smokes failed on unused vision factors; they remain in the failure record.
All42 Gemma diagnostic checkpoints have exactly-zero vision LoRA B factors, so their
vision delta-W is zero and the passed layer proofs stay valid. The revised worker
`116_mechanism_text_graph_worker.py` calls the unchanged registered HF implementation
under an explicit text-parameter projection. Only known vision/projector LoRA factors
are frozen/excluded; disconnected text factors still raise an error. Random and gradient
steps are normalized within that text subspace. Checkpoint weights stay unchanged.

Revised outputs live in `$BIDIR_OUT/results/mechanism_text_graph_v2`; the two smokes and
six production jobs use `ev_mx2_hf_` names and an independent tested enablement marker.
The scheduler and collector route those eight Gemma HF entries to revised proof checks.
The panel still contains24 production jobs, not30. Original layer and Llama HF jobs
keep their existing namespace, source proofs and queue positions.
