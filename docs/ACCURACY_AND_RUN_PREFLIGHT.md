# Manuscript accuracy and runtime preflight

Checked 2026-10-11T04:02:19.565319+00:00 (UTC). No mismatches were found in the reported
means, printed percentages, numerical macros, source hashes, corpus counts or
frozen translation thresholds. This report describes the recorded snapshot;
the background pipeline continues to update the manuscript as experiments finish.
The companion `ACCURACY_PREFLIGHT.json` records the exact document/report hashes.

## Scientific and numerical checks

- Independently replayed binary trial scores for 125 distinct campaigns,
  checking 1888 means across their reports. Repeated report checks are not
  additional independent measurements.
- Checked 542 printed percentages in the primary, full-dose and objective
  grids, including task/model/direction labels, highlighted values, missing entries
  and code-specific CFT applicability.
- Checked 317 registered numerical values and all 43 numeric
  macros used in the main manuscript. Formatting's representative 1% result agrees
  with its recorded forward/backward trials.
- Verified 739 source hashes, 21 corpus split sizes and 20 frozen
  translation threshold rows. Confirmed 91 unique model/task/training-run
  combinations, 52 severe-loss cases, matched example/step counts in
  66 training summaries and ranges for 9 repeated task orientations.
- Reviewed task/scorer implementations against the explanations, including algebra
  orientation, execution preimages, format parsing/copy guards, translation threshold
  comparability, factual aliases and the frozen SQL reverse parser. Clarified that
  the SQL keyword guard requires whole-word SELECT forward and rejects it backward,
  case-insensitively; a valid reverse question can therefore be rejected.
- Checked resolution of all 16 cited bibliography keys and read the saved primary
  abstracts of the six references central to the reversal, preservation and objective
  framing (Berglund, Zhu, Nikiema, RevThink, latent-generation and spectral-unforgetting).

No training objective, data split, correctness predicate or proof-pinned worker was
changed. The audit replays stored scores; it does not regenerate model answers,
independently relabel every answer, recompute every bootstrap interval, or certify
unrecorded historical dataset revisions.

## Runtime checks and corrections

- Verified all seven initial GPU checks through current completion receipts. The
  relations full evaluation and execution-prediction production training completed
  successfully. Code production training is running; five next phases await Slurm
  priority. 9 arm/task/model combinations still need reported evaluations.
- Verified the bytes and finite recorded training losses of 37 available
  prerequisite/completed adapters, original capability/scorer checks, recipe batch
  sizes, and one active phase per campaign.
- Runtime imports and the COMET environment are present. The SQL generator/parser
  GPU fractions sum to 0.9; scratch has about 25,048 GiB free.
  GPU admission has no artificial cap. The earlier 32 holds remain in place under
  the user's missing-table priority.
- All five controller health checks pass: heartbeat, supervisor, completed stages,
  detached execution and the dependent CPU-host renewal allocation.
- Fixed two stale test fixtures: dynamic source-backed main-result macros were
  missing from the placeholder test, and a routing fixture accidentally depended
  on the real diagnostic policy. Added six audit regression tests for duplicate
  scores, invalid binary scores, swapped directions/model labels, CFT applicability
  and highlighted table percentages.
- Full suite: **441 passed**, five library warnings. Strict PDF checks pass: seven
  body pages, 26 total pages, 126-word abstract, embedded fonts, resolved numeric
  macros/references and available assets. These checks do not imply table completion.

Re-run from the repository root:

```bash
source scripts/env.sh
python scripts/123_accuracy_preflight.py
python paper/check_arr.py --strict
python scripts/97_pipeline.py debug
pytest -q
```

The preflight is read-only with respect to experiments and scientific results. Its
live JSON report is `runs/feeder/accuracy_preflight_latest.json`. Successful smoke
runs and prerequisite checks reduce failure risk; production jobs still depend on
scheduler capacity, GPU memory and runtime conditions.
