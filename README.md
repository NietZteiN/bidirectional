# Directional collapse under one-way fine-tuning

Many NLP tasks come in pairs that are the same data read two ways: obfuscate and deobfuscate,
en→de and de→en, text-to-SQL and SQL-to-text, RDF-to-text and text-to-RDF. This project tests
whether fine-tuning on **one** direction destroys the ability the model already had in the
**other** — how general that is, what predicts it, how little reversed data prevents it, and
whether the lost capability is erased or merely suppressed.

- **The science:** [`acl2027-directional-collapse-plan.md`](acl2027-directional-collapse-plan.md)
- **The execution plan:** [`RUN_PLAN.md`](RUN_PLAN.md) — panel, budget, phases, decision gate
- **Operating rules:** [`CLAUDE.md`](CLAUDE.md) — compute, environment, and the rules that make
  a result mean something

## Layout

```
configs/     models.yaml (the panel) · domains/ (one per cell, incl. frozen thresholds)
             train/_base_lora.yaml (the ONE recipe every arm trains under)
src/bidir/   schema · arms · mixture · prompts · train · losses · engine · evaluate
             domains/  mt · sql · code · d2t · fmt · fmt_novel · exec_pred
             mech/     elicitation · alpha_scale · layer_ablate · direction_probe · sensitivity
scripts/     00_status · 10_build_domain · 15_base_gate · 20_train_pack · 30_determinism_floor
             40_probes · 50_contrasts · 51_tables · 52_figs · 53_mech_report
             slurm/    submit · pipeline_{gate,grid,mech,attrib}
tests/       121 tests; the ones that load a tokenizer must run under sbatch (see CLAUDE.md §1)
```

`src/bidir/` imports **obtune as a library** — its vLLM engine, chat-template adaptation,
sandboxed executor and provenance layer — rather than forking it.

## Setup

```bash
bash env/setup_env.sh          # training env: obtune's lock + scoring extras
bash env/setup_score_env.sh    # COMET only; it pins transformers<5
source scripts/env.sh
make check                     # tests + the dry-runs that catch a broken pipeline
```

## Running

```bash
# Data (CPU, partition `normal`)
python scripts/10_build_domain.py --domain mt_en-de
python scripts/10_build_domain.py --domain fmt_det50     # an RQ3 ladder rung

# Freeze the criterion from the BASE model, before any tuned model is scored
python scripts/15_base_gate.py --domain mt_en-de --model llama32-3b --write

# The September decision gate: 4 cells x 4 arms, ~20 GPU-h
python scripts/slurm/pipeline_gate.py
python scripts/50_contrasts.py --gate        # the pass rule is code, not a reading

# The grid, then mechanism and attribution
python scripts/slurm/pipeline_grid.py --tier small
python scripts/slurm/pipeline_mech.py --floor
python scripts/slurm/pipeline_attrib.py

python scripts/00_status.py                  # what exists, what is queued, what is left
```

## The arms

`sft` is forward-only — the collapse. Every `mix{1,5,10,25,50}` replaces that share of forward
pairs with **their own reversal**, so instances, sequence tokens and optimizer steps stay
matched to `sft` and only the direction share varies. `replay` replaces the same share with
generic instruction data, which separates directional loss from ordinary forgetting. `rev` is
the reverse ceiling and the kill-gate: if it is near zero, the direction is not learnable and
no other arm's null is interpretable. `flip` and `fwd2x` are the doubled-budget references.
`relearn{k}` measures recovery from a collapsed model, read against `fmt_novel` — an invented
transform the model never had, which is what turns "fast recovery" into evidence of
suppression rather than an impression.

## The one rule that matters most

**Nothing that is scored may have been trained on.** The eval-set check runs at build time on
content, with a per-domain notion of identity, and it fails the build rather than warning. It
has already caught a real leak — the only Hub copy of Spider carrying the sqlite databases
pools the dev set into train.
