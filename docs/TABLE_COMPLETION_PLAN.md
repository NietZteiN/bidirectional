# Calculate the missing tables first — Amendment 57

The user's clarification supersedes the prose-only interpretation of Amendment56.
Complete unmeasured numerical table entries first. New table work is authorized; the
32 previously held jobs remain held so unrelated experiments cannot consume priority.

The seven fixed seed17 campaigns cover twelve missing arm/task/model combinations:

| Table target | Missing arms | Work |
|---|---|---|
| Relations / Llama3B | mix1, mix5, mix10 | Reuse three saved adapters; fresh full shared evaluation |
| Execution / Llama3B | mix1, mix5, mix10 | GPU smoke, train three arms, fresh full shared evaluation |
| Code / Llama3B | unlikelihood, round-trip | GPU smoke, train both arms, evaluate with CFT and CE controls |
| English→German / Llama3B | round-trip | GPU smoke, train, evaluate with unlikelihood and CE controls |
| English→German / Llama8B | round-trip | Same protocol |
| SQL / Llama3B | unlikelihood | GPU smoke, train, evaluate with round-trip and CE controls |
| SQL / Llama8B | unlikelihood | Same protocol |

Each campaign reports both directions and regenerates every displayed applicable arm
with a fresh base. Twelve missing arm combinations yield 24 primary percentage entries;
duplicate legacy table positions are refreshed from the same verified campaigns.
Low-dose relation/execution cells are explicitly exploratory below-batch extensions,
not evidence for the original registered knee. Only seed17 is added here.

The six omitted mixed-task normalizations can be calculated immediately from retained
trials; all null/adverse cells are retained. The signed ratio is not a recovery verdict
outside collapse cells. Zero denominators are mathematically undefined.

The user chose to keep CFT code-specific. Its four translation/SQL arm combinations
are explicitly not applicable, and no paired-judgment extension is queued. Failed capability gates
likewise remain real boundaries; the queue never fabricates tuned scores after a failed gate.

`scripts/119_table_completion.py` admits one phase at a time per cell. New training requires
four real GPU steps with positive objective exposure; production generation requires a
shared-pass generation smoke. Existing valid adapters are reused. Receipts pin data,
code, model/recipe and manifests. At most three attempts per phase are allowed; a code
or scientific failure stops that phase for diagnosis. No historical artifact is deleted.

The detached controller selects this exclusive mode through `configs/table_completion.json`.
It runs the completion scheduler, checked analysis and table/PDF refresh after logout and
host renewal. The broad runner, watchdog, old shell queue and capacity release are skipped.
It does not release the 32 previous holds. Disabling this mode or releasing those holds
requires a subsequent user instruction. `runs/feeder/table_completion_status.json` records
live progress; `paper/TABLE_COMPLETION_AUDIT.json` records remaining table positions.

## Admission and debug checkpoint — October 10

Seven first-phase jobs (451313–451319) are submitted and pending Slurm priority;
GPU validation has not yet run. The relation job checks generation from reused adapters;
the other six jobs check new training objectives/doses before production. All 32 earlier
jobs remain held. The controller passes all five health checks, including detached stdin
and CPU-host renewal, and 43 targeted regression tests pass. The rebuilt manuscript has
eight body pages, twenty total, resolved numeric macros/references and embedded fonts.
Fourteen signed mixed-task normalizations are calculated. Twelve arm combinations remain
unmeasured: 24 main-table percentages plus nine duplicate legacy table positions.
