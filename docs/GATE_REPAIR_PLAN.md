# Versioned blocked-cell repairs

Registered October 9, 2026, before new GPU diagnostics (Amendment51).
Executable registration: `configs/gate_repair.json`. This extension follows inspection
of failed originals and is exploratory. Original scorers, gates and trials remain intact.

| Diagnostic | Observed problem | Separate repair | Admission |
|---|---|---|---|
| Python/C++ | Valid Gemma12B `L`/`LL` integer literals rejected by bounded grammar | `py_cpp_typed_v2` accepts signed suffixes; bounded interpreter and compiled execution must agree on all33 inputs | Exact compute-local oracle, then fresh1000-row gate |
| Serialization | Gemma12B XML fragments omit root; scalar types change | `fmt_contract_v2` supplies root/item/type contract; original structural scorer | Exact oracle and fresh gate; matching objective/control trainer smokes for Gemma12B CL |
| Units | Output format and conversion errors | `units_contract_v2` states exact format/conversion rules; unchanged rational scorer | Exact oracle and fresh gate |
| Boolean logic | Formula echoes and incorrect truth-table/minterm mapping | `logic_contract_v2` states assignment order and construction; unchanged equivalence scorer | Exact oracle and fresh gate; genuine capability failures may remain |
| OPUS transfer | Forward source copies could pass the original directional MT predicate | `mt_noecho_v2` excludes copies in both directions; original model-specific COMET thresholds | Fresh engineering smoke and fresh same-pass base gate before any tuned generation |
| WebNLG | Frozen extractor paraphrases predicates and misses reference facts | `d2t_schema_v2` gives352 predicate labels derived only from training; fixed Unicode diacritic/predicate-space normalization | Frozen Granite40-gold oracle must pass before the Llama3B base gate; no test-target hints |
| Legacy domains | Original small-model gates fail | Unchanged automata, diacritics, fmt_det75/50/25 and coverage at Llama8B/Gemma12B | Original oracle tolerance, fresh gate; failed cells remain boundaries |

New scorer/prompt cells require gold strict1.0 and echo/empty/garbage strict0.0 in both
directions on40 cases. Legacy criteria retain their original oracle tolerance (gold≥.95,
negative controls≤.02). Base gates retain rate≥.10 and format-failure≤.15 independently in
both directions, seed17,1000 sampled test pairs or every pair when fewer are available.
Coverage has298 test pairs. No threshold tuning, best-prompt selection or substitution of
old base rates is allowed. Technical failures retry under the existing bounded watchdog;
scientific failures complete as diagnostics and never trigger endless retries.

The four repaired domains are gated at Llama3B/Gemma4B/Llama8B/Gemma12B. Passing cells
receive base/sft/rev/mix50/replay fresh-pass pilots and seeds42/1234 after seed17 passes
implementation validity, irrespective of the sign of its effects. The exception is
fmt_contract_v2/Gemma12B: its three seeds use the existing ten-system contrastive panel,
with five primary objective/control contrasts, λ=.1, temperature=.1 and four fixed negatives.
Objective and extra-CE control each require a new two-step real-model trainer smoke.
Its negative pool is the original fmt training split; paired text/subtask content must match
the versioned training split exactly before training. Versioned IDs determine fixed negative
selection. New wrappers isolate this alias from existing training processes, and each adapter
records the new protocol/data/code/negative-pool and ordinary training-manifest hashes.
No simultaneous domain-pilot pack targets the same fmt_contract_v2/Gemma12B adapters.

WebNLG overlap auditing additionally uses the new canonical triple set, rather than
reference wording. Neither training nor validation overlaps the held-out test under this
key. Two triple sets recur across training/validation and are recorded in the build report;
pair IDs remain disjoint, and no validation checkpoint selection is performed.

Passing unchanged legacy gates similarly admit the four-arm pilot at all three seeds.
The WebNLG schema variant admits Llama3B and three seeds only after its extractor oracle
and base gate pass. The underdetermined fmt_det00 inverse remains a capability boundary.
OPUS reruns all original paired systems at all three seeds for Llama3B/Gemma4B, including
failed fresh base gates, in its separate result namespace. Existing original adapters are
reused only for transfer; revised prompt/domain tasks train new adapters.

Worker: `scripts/109_gate_repair_worker.py`; admission: `scripts/108_gate_repair.py`.
Proofs/results: `$BIDIR_OUT/results/gate_repair_v1`; feeder report:
`runs/feeder/gate_repair_status.json`. Successful job status, current source/data hashes,
complete trial and oracle hashes are required. Original39 publication jobs remain a separate
completed panel. New results are exploratory additions, not replacements for original failures.
The detached controller launches ready diagnostics and subsequently valid pilots automatically;
GPU caps remain disabled. Trials, criteria, datasets and adapter manifests are mirrored locally
and included in the review bundle; weights are excluded.
