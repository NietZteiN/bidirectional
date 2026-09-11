# Run plan — Directional collapse (ACL 2027)

*Written 2026-09-09 from an audit of `obtune/`, the cluster, and the Hub. Companion to
`acl2027-directional-collapse-plan.md` (the science); this file is the execution.*

## 0. What the audit changed

| plan doc says | what is actually true | consequence |
|---|---|---|
| Qwen2.5-Coder / Qwen3 lead the panel (§5); code domain is "done" | `obtune/configs/models.yaml` (2026-09-09): **no Chinese-origin model for any paper result**; Qwen `role: barred`, "cannot be used on this cluster". Every workshop-paper number is Qwen. | New panel (§2). Code domain is **re-run** on it — cheap, the pipeline exists. |
| ~1 GPU-h per small run, ~3 per 7B; ~1,200 GPU-h total (§6) | Measured on H200: **7B LoRA adapter = 18 min** (4.7k rows × 3 ep). 7k rows ≈ 30 min at 8B, ≈ 12 min at 3–4B. | Full version ≈ **390 GPU-h**, minimum ≈ **100** (§4). |
| compute is the constraint | **Concurrency is the constraint**: 4 running jobs per user on `h200` (partition QoS `juno`; your obtune and transcoders queues are pending on `QOSMaxJobsPerUserLim` right now). Cluster at 46/52 H200s busy. | Pack many adapters per job; chain with `afterok`; spill ≤3B work to `h100`/`a30` (separate caps). |
| — | vLLM multi-LoRA eval **works** (`log/setup/2026-08-30_vllm-unblocked.md`; the CLAUDE.md §2 verdict is stale) | One eval pass per (domain, model, seed), all arms, ~15 min. |
| 8.8 GPU-h figure needs confirming (§13) | It was an A6000 number from the old host; irrelevant on H200. | Drop it; use the measurements above. |
| — | Env `obtune-cu129` lacks `lm_eval`, `sacrebleu`, `unbabel-comet`, `sqlglot`, `evaluate`, `nltk` | Clone the env, add scoring deps (§3.1). Do not touch obtune's lock. |
| — | All datasets on the Hub, ungated or auto-gated; Llama-3.2-3B, Gemma-3-1B, OLMo-2, COMET-22 downloadable with the on-disk token; compute nodes have internet | No data blockers. Spider **sqlite DBs + test-suite** are on GitHub, not the Hub. |
| — | `node` not installed on juno | Code domain is Python-only. |

## 1. Decisions (defaults stated; the gate does not wait on D3–D6)

- **D1 Panel** (blocking): see §2. Alternative small model if OLMo-2-1B fails its base gate: Phi-3.5-mini (on disk).
- **D2 Where code lives**: new package `src/bidir/` in this repo, importing obtune as a library (`Engine`, `prompts.template_mode/adapt_messages/to_trl_example`, executor, `submit.py` pattern). Not a fork, not an extension: obtune's quarantine lint greps its own `src/`+`scripts/` for raw reads and its CLAUDE.md governs another paper.
- **D3 Language pairs**: en–de (full arms, both training directions) + en–zh (core arms, both directions; script change makes off-target trivially detectable).
- **D4 Execution prediction** as sixth domain: yes, after the gate (CRUXEval cached; obtune executor exists).
- **D5 QoS**: stay on the default; `juno-pri` (8 slots, priority 200000) jumps every other user and obtune's own `submit.py` deliberately avoids it.
- **D6 Env**: cloned venv `envs/bidir-cu129` = obtune's lock + scoring extras.

## 2. Panel (non-Chinese, three lineages)

| tier | model | lineage | on disk | template | notes |
|---|---|---|---|---|---|
| small (full arms, 3 seeds) | Llama-3.2-3B-Instruct | Meta | no (gated, access verified) | system | |
| small | Gemma-3-4B-it | Google | **yes** | folds system into user | `Gemma3ForConditionalGeneration`; needs `peft_exclude_modules: [vision_tower, multi_modal_projector]` (obtune already does this for 12B) |
| small | OLMo-2-0425-1B-Instruct | AI2, open data | no (ungated) | system | answers the contamination question (§4.2 of the plan); base gate may fail on hard domains → Phi-3.5-mini fallback |
| large (core arms, 2 seeds) | Llama-3.1-8B-Instruct | Meta | **yes** | system | |
| large | Gemma-3-12B-it | Google | **yes** | folds | same exclusions |
| base replicate (one domain) | Llama-3.1-8B (pretrained) | Meta | **yes** | `plain_v1` path exists in obtune | |
| code continuity (optional) | CodeLlama-7B-Instruct | Meta | **yes** | system | ties to the FSE paper's panel |

Gate every model the way obtune does before training an adapter on it: untuned forward/reverse rates and `format_fail` ≤ 0.15, truncation at 2048 ≈ 0 on 1,500 rows, loss-mask gate on one batch, adapter-applied assertion on the first eval.

## 3. What gets built

### 3.1 Environment (login node, ~1 h)
```
uv venv /work/jvl210002/migration/envs/bidir-cu129 --python 3.12
uv pip install -r obtune/env/lock-obtune.txt            # same 222 pins
uv pip install torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129
uv pip install lm-eval sacrebleu unbabel-comet sqlglot evaluate nltk
```
Then one `a30` smoke job: vLLM engine start + one LoRA generate (reuse `obtune/scripts/vllm_smoke.py`). Verify `lm_eval --model vllm` runs with `lora_local_path`.

### 3.2 Package layout
```
bidirectional/
  configs/
    models.yaml                 panel entries in obtune's schema (hf_id, n_layers, batch shape, render_mode, peft_exclude_modules)
    domains/{code,mt,sql,d2t,fmt,exec}.yaml   source, pair construction, eval set, criteria, thresholds τ (frozen from base distributions)
    grids/{gate,small,large,mech,attr}.yaml   which (domain, model, seed, arms) cells exist
  src/bidir/
    schema.py       PairInstance{pair_id, domain, subtask, side_a, side_b, split, meta}. Direction-agnostic: `fwd` reads a→b, `rev` reads b→a, no field swap (obtune's invariant).
    arms.py         registry, one place that says what each arm IS (port of obtune/src/obtune/srh/arms.py):
                    sft fwd2x rev mix{1,5,10,25,50} flip replay mixedtask fullft relearn{10,50,200,1000} cft unlikelihood roundtrip
    mixture.py      direction_mix partitioned by pair_id (port of srh/dataset.split_directions); replay = replace share with tulu-3-sft-mixture rows;
                    mixedtask = forward pairs as 20 % of a five-task SFT set (the other four = the other domains' forward pairs)
    prompts.py      ONE shared system prompt for both directions (flipsym showed no difference); per-domain user turns; rev-train prompt == rev-eval "simple" prompt, asserted
    train.py        TRL SFTTrainer, LoRA r32/α64/dropout .05, lr 1e-4 cosine, batch 64, 3 ep, completion_only_loss, packing=False (port of cft/train.py minus code-specific gates)
                    flags: --full-ft, --init-adapter (relearn-k), --loss {ce,unlikelihood,roundtrip}
    evaluate.py     one vLLM pass: systems × directions × strategies → trials.jsonl + summary.json; adapter-effective guard AFTER rows are written (obtune's lesson)
    domains/        load(), build_pairs(), user_prompt(direction, inst), score(direction, output, inst) → {strict, continuous, echo, off_target, format_fail}
      code.py       obtune corpus (Python; L1b/L1r/L2/S1/S2 pairs) + obtune.exec
      mt.py         news_commentary en-de / en-zh (europarl backup) → 7k; FLORES-200 devtest 1,012; sacrebleu chrF++/BLEU + COMET-22; lang-id for off-target
      sql.py        Spider train → 7k; dev 500; sqlite execution; reverse = SQL→NL scored by round-trip through a frozen NL→SQL parser (a model NOT in the panel — Granite-3.1-8B, on disk) + BLEU/BERTScore secondary
      d2t.py        WebNLG 2020 v3.0 (`webnlg-challenge/web_nlg`, release_v3.0_en) → 7k; test set; forward = round-trip triple recall via frozen extractor + chrF++; reverse = exact triple-set match
      fmt.py        synthetic: JSON↔YAML, JSON↔XML, MD-table↔CSV, 7k/500; parse ∧ structural equality; lossy ladder drops {0,10,50,100}% of leaf values
      exec.py       CRUXEval output/input prediction via obtune.exec
    probes.py       lm-eval (MMLU 5-shot, GSM8K, IFEval) with vllm backend + LoRA; MBPP+ for code via obtune.forgetting
    stats.py        cluster bootstrap by pair_id, paired contrasts within one pass, cross-pass floor (port of scripts/srh/24_contrasts.py)
    mech/           alpha_scale.py, relearn.py, layer_ablate.py, direction_probe.py, sensitivity.py
  scripts/
    10_build_domain.py --domain X                    CPU (`normal`)
    15_base_gate.py --model M --domain X             untuned rates → configs/domains/X.yaml thresholds
    20_train_pack.py --domain --model --seed --arms  ONE GPU job trains its arms sequentially; skips adapters that exist (walltime-safe)
    30_eval_pass.py --domain --model --seed          ONE GPU job, all arms, one vLLM pass
    40_probes.py --adapters ...
    50_contrasts.py  51_tables.py  52_figs.py
    slurm/submit.py (obtune's, with BIDIR_ROOT)  pipeline_gate.py  pipeline_grid.py
  tests/            rev prompt == eval prompt; mix is pair-disjoint; loss mask −100 on prompt; eval ∩ train = ∅ per domain; scorer fixtures per domain
```

**Packing rule.** A job is (domain, model, seed) and trains every arm in that cell back-to-back, then a dependent job evaluates the cell in one pass. At 3B that is 13 adapter-units × 12 min ≈ 2.6 h (ask 6 h). The small grid becomes ~60 train jobs + ~60 eval jobs, not 700.

### 3.3 Correctness rules inherited from obtune (write them into tests, not prose)
1. Train/eval prompts byte-identical, rendered through the same `template_mode` path.
2. Thresholds τ set from the **base** model's score distribution before any tuned model is scored; written to the domain config with the date.
3. One eval pass per table; contrasts only within a pass; never quote a cross-pass difference below 0.5 pp (vLLM batch nondeterminism, measured at 6–8 % of generations).
4. Overlap check per probe, written down before it runs (the HumanEval+ contamination cost the workshop paper a claim).
5. Adapter-effective assertion on every eval; rows written before the guard raises.
6. All outputs on `/work` (compute-node `/tmp` is node-local); manifests + resolved configs committed, adapters/trials gitignored.

## 4. Budget (measured rates: 3–4B 0.2 h/adapter · 8B 0.5 h · 12B 0.85 h · eval pass 0.25/0.5 h · probes 0.3/0.5 h)

| block | cells | adapters | GPU-h |
|---|---|---|---|
| **Gate** | MT en→de, MT de→en, SQL × Llama-3.2-3B × s17; arms sft mix5 mix50 rev | 12 | **~5** |
| Small grid, full arms | 6 cells (code, MT en-de ×2, SQL, D2T, FMT) × 3 models × 3 seeds × 13 units | 700 | 140 |
| Small grid, core arms | MT en-zh ×2 × 3 × 3 × 4 | 72 | 14 |
| FMT lossy ladder | 3 variants × {sft, mix50} × 3 models × s17 | 18 | 4 |
| Exec prediction (D4) | 1 cell × 3 × 3 × 13 | 117 | 23 |
| Eval passes, small | 81 | — | 20 |
| Large core | 5 domains × {8B, 12B} × {sft, mix5, mix50, replay, rev} × 2 seeds | 100 | 68 |
| Eval passes, large | 20 | — | 10 |
| Full-FT | {sft, mix5} × 3 models × 5 domains × s17 | 30 | 15 |
| Relearn-k | 4 k × 5 domains × 2 models, from sft | 40 tiny | 6 |
| Mechanism 1–3 | elicitation ×4 strategies, α-ladder ×6 | — | 25 |
| Mechanism 4–6 | layer ablation, probes, sensitivity | — | 20 |
| Attribution | cft (code, 3 models), unlikelihood + roundtrip (MT/SQL, 2 models × 2 seeds) | 19 | 8 |
| General probes | core arms × 5 domains × 5 models × s17 (+ base once per model) | ~150 | 40 |
| **Full version** | | | **≈ 390** |
| **Minimum version** (3 domains, 2 families, 3 seeds small, 1 seed 8B, mech 1–3, cft only) | | | **≈ 100** |

Wall-clock: 4 slots on `h200` shared with your other two queues → plan on ~2 effective slots ≈ 48 GPU-h/day → full version ≈ 8 compute-days, **3–4 calendar weeks** with queue waits, spread over Oct–Nov. `h100` (4 more slots; g-06-01's 47 GB MIG slices train 3B fine) and `a30` (24 GB; ≤3B only) are separate caps and should carry the small-model packs.

## 5. Phases and job lists

### Phase 0 — infrastructure (Sept 10–19, mostly CPU)
| # | what | where |
|---|---|---|
| 0.1 | env clone + extras; vLLM+LoRA smoke; lm-eval smoke | login + 1 `a30` job |
| 0.2 | download Llama-3.2-3B-Instruct, OLMo-2-1B/7B-Instruct, Gemma-3-1B-it (spare), COMET-22; `models.yaml` entries; `template_mode` check per model | login |
| 0.3 | datasets: news_commentary (en-de, en-zh), europarl, flores_plus, spider + `spider_data.zip` DBs + `taoyds/test-suite-sql-eval`, web_nlg v3.0, tulu-3-sft-mixture (replay), IFEval/GSM8K/MMLU | login / `normal` |
| 0.4 | `schema/arms/mixture/prompts/train/evaluate` ports + tests | — |
| 0.5 | domain modules: `mt`, `sql`, `code` (gate domains + the known-positive control) | — |
| 0.6 | `10_build_domain` for mt, sql, code → pair files, split manifests, overlap reports | `normal` |
| 0.7 | `15_base_gate` for Llama-3.2-3B on mt/sql/code → τ frozen, written to configs | 1 `h200` job, ~1 h |

### Phase 1 — decision gate (submit ~Sept 20, read ~Sept 26)
```
J1  train_pack mt_en-de  llama32-3b s17  arms=sft,mix5,mix50,rev     h200  6h
J2  train_pack mt_de-en  llama32-3b s17  arms=sft,mix5,mix50,rev     h200  6h
J3  train_pack sql       llama32-3b s17  arms=sft,mix5,mix50,rev     h200  6h
J4  train_pack code      llama32-3b s17  arms=sft,mix5,mix50,rev     h100  6h   (the known-positive control on the new panel)
J5–8  eval_pass for each, afterok                                    h200  2h
J9  probes IFEval+GSM8K on base,sft,mix5 for the four cells         h200  3h
J10 contrasts + gate report (CPU)                                    normal
```
**Pass rule** (pre-registered): in ≥1 NLP cell, `sft − base` on strict reverse ≤ −50 % relative **and** IFEval `sft − base` > −5 pts **and** `rev` strict reverse ≫ 0 (kill-gate: if `rev` ≈ 0 the direction is not learnable here and no null is interpretable). If it passes → Phase 2. If not → the boundary-conditions paper; reset before more compute.

### Phase 2 — small grid (Oct; first week light)
- Build `d2t`, `fmt` (+ ladder), `exec`; base gates for Gemma-3-4B and OLMo-2-1B on every domain; freeze τ per model.
- Pre-register the arm list and primary contrasts (`sft−base`, `mix5−sft`, `replay−sft`, `mix50−flip`) — commit the file before submitting.
- Submit `pipeline_grid.py --tier small`: ~60 packed train jobs + 60 eval jobs, seeds {17, 42, 1234}, spread across h200/h100/a30 by model size. Then the en-zh core cells and the FMT ladder.
- Probes on core arms, seed 17.

### Phase 3 — large core + mechanism + attribution (Nov)
- `pipeline_grid.py --tier large`: 8B + 12B, core arms, seeds {17, 42}.
- Mechanism 1–3 (behavioral, cheap): elicitation ladder, α-scaling, relearn-k (with the never-had contrast: a novel synthetic FMT transform). Then 4–6 as time allows.
- Attribution: `cft` on code (port obtune's pools), `unlikelihood` (Zan et al.) and `roundtrip` on MT/SQL, each against direction-matched SFT at matched sequence tokens.
- Full-FT arms; base-model replicate on one domain.

### Phase 4 — writing (Dec–early Jan)
Tables/figures generated from `trials.jsonl` + `contrasts.json` (never hand-typed; obtune's `25_fig_dose.py` pattern). Third-family fill-ins only if the queue allows.

## 6. Risks specific to this cluster
- **Concurrency cap is shared across your three projects.** Decide who gets the slots each week; the packing rule is what makes 4 slots enough.
- **2-day walltime**: packs skip existing adapters, so a killed pack resumes by resubmission.
- **Spider execution** needs the sqlite DBs and the test-suite repo from GitHub (~1 GB), not the Hub.
- **Gemma-3-4B is multimodal**: same `peft_exclude_modules` as 12B, load text-only.
- **OLMo-2-1B base ability** may be ~0 on SQL/D2T → kill-gate per (model, domain); swap to Phi-3.5-mini, don't drop the lineage.
- **Text-domain reverse criteria are noisier than execution**: τ from base distributions, continuous metrics beside strict rates, always.
- **The ARR policy on extending non-archival workshop papers** — check before October.

## 7. Phase 0 — what is built (2026-09-09 / 09-10)

Everything below is on disk and tested. Nothing here needs a GPU; the grid is submittable the
moment slots free up.

| | item | state |
|---|---|---|
| 0.1 | `envs/bidir-cu129` (training) + `envs/bidir-score` (COMET) | built |
| 0.2 | Llama-3.2-3B, OLMo-2-1B, OLMo-2-7B, COMET-22 downloaded; `configs/models.yaml` — 9 entries, every `n_layers`/`hidden_size` from the downloaded `config.json`, every `render_mode` measured on the real tokenizer | done |
| 0.3 | all datasets cached | done |
| 0.4 | `src/bidir/`: `config` `schema` `prompts` `arms` `mixture` `train` `losses` `collators` `engine` `evaluate` | written |
| 0.5 | domains: `mt` `sql` `code` `fmt` `d2t` `exec_pred` | written |
| 0.6 | **12 domain cells built — 83,099 pairs, zero eval leakage** | done |
| 0.7 | `mech/`: `elicitation` `alpha_scale` `layer_ablate` `direction_probe` `sensitivity` | written |
| 0.8 | scripts `10`–`52` + `slurm/{submit,pipeline_gate,pipeline_grid}`; every tier dry-runs | done |
| 0.9 | **111 tests** | passing |

### The 12 built cells

| cell | forward → reverse | train / val / test |
|---|---|---|
| `mt_en-de`, `mt_de-en` | en↔de, both training directions | 6,500 / 500 / 500 |
| `mt_en-zh`, `mt_zh-en` | en↔zh, both training directions | 6,500 / 500 / 500 |
| `sql` | NL↔SQL, Spider | 6,500 / 500 / 500 |
| `code` | obfuscate↔deobfuscate, 5 transforms | 6,500 / 500 / 300 |
| `d2t` | RDF↔text, WebNLG v3.0 | 6,500 / 500 / 500 |
| `fmt` | JSON↔YAML/XML, MD-table↔CSV | 6,500 / 500 / 500 |
| `fmt_det{75,50,25,00}` | the RQ3 invertibility ladder, rungs spread by the share of instances whose inverse is determined | 6,500 / 500 / 500 each |
| `exec` | output↔input prediction, CRUXEval | 499 / 100 / 200 |

### Grid tiers, dry-run against the budget

| tier | cells | arms | adapter-units | plan estimate |
|---|---|---|---|---|
| `small` | 54 | 11 (`full`) | 702 | 700 |
| `small_zh` | 18 | 5 (`core`) | 90 | 72 |
| `large` | 20 | 5 (`core`) | 100 | 100 |
| `ladder` | 9 | 2 | 18 | 18 |
| `exec` | 9 | 11 | 117 | 117 |
| `relearn` | 10 | 4 | 6 | 6 |
| `fullft` | 15 | 2 | 60 | 30 |

### Seven things building it changed

1. **Spider's Hub mirror is not Spider's split.** `prem-research/spider` is the only Hub copy
   with the 169 sqlite databases, and its `train.json` pools dev: all 20 dev databases appear
   in train, with 384 exact duplicate (question, query) pairs. The build's leakage check caught
   130 landing in the eval set. Questions now come from `xlangai/spider` (the official
   database-disjoint split), the mirror supplies databases only, and disjointness is asserted
   at build time.
2. **Corpus size is 6,500 + 500.** Spider's official train split is exactly 7,000 rows and the
   val slice comes out of it, so every domain matches Spider and budgets compare exactly.
3. **FLORES-200 needs no gate request.** `flores_plus` and `facebook/flores` both 403;
   `haoranxu/FLORES-200` is ungated and carries the same 1,012 devtest sentences.
4. **COMET cannot share the training env** — it pins `transformers<5` against the lock's
   5.14.1. Own venv, called as a subprocess.
5. **The leakage check has to be domain-aware.** Two different CRUXEval programs can share an
   (input, output) pair, so the generic `side_a + side_b` key produced false positives on
   `exec`. Domains now supply a `content_key`; `exec` includes the program.
6. **The login node caps virtual memory at 8 GB.** Model loads, and even scipy under
   transformers, die there and work in a job. Verification runs as a SLURM job.
7. **`json-yaml` needs the echo guard to be a valid task at all.** JSON is valid YAML, so an
   echoing model parses *and* compares structurally equal. Echo is a first-class column in
   every scorer for this reason, and every strict criterion conjoins "not an echo".

## 8. What is built, by phase

Phase 0 is complete. The phases below are **built and dry-running**; they are waiting on GPU
slots, not on work.

| phase | infrastructure | entry point |
|---|---|---|
| **1** decision gate | 4 cells x 4 arms, packed, `afterok`-chained, pass rule in code | `slurm/pipeline_gate.py` → `50_contrasts.py --gate` |
| **2** small grid | 8 tiers (`small`, `small_zh`, `ladder`, `exec`, `large`, `relearn`, `fullft`, `base_replicate`), placement by model size across three partitions | `slurm/pipeline_grid.py --tier X` |
| **3a** mechanism | experiments 1-6, floor and upside submitted separately so a stalled probe cannot hold up the result that decides RQ5 | `slurm/pipeline_mech.py --floor` / `--all` → `53_mech_report.py` |
| **3b** attribution | three objectives, each paired with the `mix` arm supplying the same directional content and none of the method | `slurm/pipeline_attrib.py` |
| **4** writing | tables and figures generated from `trials.jsonl`; provenance table; pre-registration committed before any adapter was trained | `51_tables.py`, `52_figs.py`, `paper/NUMBERS.md`, `PREREGISTRATION.md` |

Supporting: `00_status.py` (what exists, what is queued, what is left), `30_determinism_floor.py`
(the cross-pass floor every "never quote finer than" rule cites), `Makefile` (`make check` runs
the tests plus every pipeline dry-run), `CLAUDE.md` (operating rules), `README.md`.

### Two things built after the first pass

- **`fmt_novel` — the never-had control.** Mechanism experiment 3 compares relearning from a
  collapsed model against learning a capability the model never had; without the second curve
  there is no scale to read the first against, and no experimental contrast with the
  reversal-curse literature. It is `fmt`'s own generator and documents in an **invented**
  encoding (reversed keys, sigil delimiters) that no pretraining corpus contains, so the base
  model is at floor by construction rather than by luck. Lossless and easy to learn on purpose:
  if it were hard, a slow curve would be about difficulty rather than novelty.
- **The `cft` arm.** The workshop paper's worked example, reading obtune's equivalence pools.
  Its realised mixture is 6,500 forward + 7,384 `pos` + 5,612 `neg` at **reverse_share 0.0** —
  the attribution point in one number: instances added, no reverse exposure added.

## 9. The domain set

**18 cells.** The six original domains, the `fmt_det*` ladder and `fmt_novel`, plus three added
2026-09-10 on the argument in [`docs/CANDIDATE_DOMAINS.md`](docs/CANDIDATE_DOMAINS.md):

| cell | what it closes |
|---|---|
| `algebra` (#56) | the paper claimed generality across paired tasks with **no formal domain** |
| `diacritics` (#66) | the only domain whose forward direction is trivial, so a reverse collapse cannot be capacity-spend |
| `automata` (#30) | a **determined but computationally hard** inverse — the axis `fmt_det*` cannot reach |

`automata` is the one that changes a prediction rather than adding a data point. RQ3's
"invertibility" was two properties run together: whether the inverse is *determined*, and
whether it is *findable*. `fmt_det*` varies the first; `automata` holds it fixed at "determined"
and varies only the second. The sharpened prediction — a reverse dose buys recovery against
computational hardness but not against missing information — is in
[`PREREGISTRATION.md`](PREREGISTRATION.md) Amendment 4.

**Two catalogue entries were already in the grid and cost nothing.** obtune's `L2` condition is
#4 minification, and the structural (`S1`, `S2`) versus renaming (`L1b`, `L1r`, `L2`) contrast is
a within-domain invertibility test already being run. Both should be named in the paper.

Budget with the additions: small tier 81 cells, **~391 GPU-h** of training across all tiers.

### Still on the menu, not in the plan

[`docs/TASK_CATALOGUE.md`](docs/TASK_CATALOGUE.md) holds the other 77.
[`docs/CANDIDATE_DOMAINS.md`](docs/CANDIDATE_DOMAINS.md) says what to leave out and why —
weak verifiers blur the effect, and per-language build-and-test harnesses (#1 compilation,
#13 cross-language translation) are the right *next* paper rather than this one.

## 9b. Extending it further

[`docs/TASK_CATALOGUE.md`](docs/TASK_CATALOGUE.md) holds 80 candidate paired tasks.
[`docs/CANDIDATE_DOMAINS.md`](docs/CANDIDATE_DOMAINS.md) picks the three that close a hole this
paper's argument actually has — a provably hard inverse (#30 cellular automata), a formal
domain (#56 expansion ↔ factorization), and a task whose forward direction is trivial so a
collapse cannot be capacity-spend (#66 diacritic restoration) — at ~23 GPU-h each, and says what
to leave out and why.

**None of them before the gate.** The gate may reframe the paper entirely, and each domain is
23 GPU-h spent against an unchecked hypothesis. Any addition is declared in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before its results are seen.

Two catalogue entries are already inside `code` and cost nothing: `L2` is minification (#4), and
the structural-versus-renaming contrast is a within-domain invertibility test already in the
grid.

## 10. Immediate next actions

**Nothing further can be built without GPU slots.** In order, when they free:

1. Base gates → freeze τ: `15_base_gate.py --domain {mt_en-de,mt_de-en,sql,code} --model llama32-3b --write`
2. `30_determinism_floor.py --domain mt_en-de --model llama32-3b` — the floor every later claim cites
3. `slurm/pipeline_gate.py`, then `50_contrasts.py --gate`
4. On a pass: `pipeline_grid.py --tier small`, then the rest in budget order
