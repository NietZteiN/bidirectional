# Results progress — October 11, 2026 morning CDT

Six of the seven prioritized table-completion campaigns finished overnight.
Eleven of twelve missing arm/task/model combinations are now reported, and the
primary base/SFT/1%/5% table is fully populated. SQL/Llama3B unlikelihood is the
only missing combination; its full evaluation is running as job 451562 after a
diagnosed host-memory failure and an increase from 64 to 128 GiB. No scientific
setting changed. Earlier unrelated jobs remain held.

## Newly completed objective comparisons

Code transformation, Llama3B; all methods evaluated on the same examples:

| Method | Forward success (%) | Backward success (%) |
|---|---:|---:|
| Forward SFT | 82.1 | 3.3 |
| 5% reversal | 84.1 | 75.3 |
| Code-specific CFT | 83.9 | 6.5 |
| Unlikelihood | 84.5 | 0.0 |
| Round-trip | 81.6 | 33.4 |

Here, the small reversal mixture has much higher backward success than the
specialized objectives while retaining forward success. This is a measured task
comparison, not evidence of universal dominance.

The results vary across domains. For SQL/Llama8B, the 5% mixture scores 71.9%
forward and 57.8% backward; round-trip scores 71.1% and 58.6%. For English→German
on Llama8B, the 5% mixture scores 73.5%/69.9%, compared with 72.4%/40.6% for
round-trip and 74.1%/0.1% for unlikelihood. Corresponding Llama3B backward scores
are 68.5%, 67.1% and 0.0%.

## Newly completed small-corpus comparisons

Relations: forward/backward success is 86.7%/71.1% for SFT, 88.1%/73.3% for
1% reversal and 87.4%/75.6% for 5%. Execution prediction: 29.0%/13.0% for SFT,
27.0%/14.0% for 1% and 30.5%/17.5% for 5%. These exploratory small-corpus
experiments show modest gains; they do not reproduce the dramatic directional
loss in formatting or code.

The refreshed audit independently replays stored binary scores and finds no
mismatches in 562 printed table percentages or 331 registered numerical values.
It also verifies 758 source hashes, 21 corpus split sizes and 20 frozen translation
threshold rows. `RESULTS_PREFLIGHT_2026-10-11.json` binds these checks to the exact
document/report hashes. Strict PDF checks pass: seven body pages, 26 total pages,
126-word abstract, embedded fonts and resolved references/numerical macros.

The paper will refresh automatically after the final SQL evaluation succeeds.
Success remains conditional on the running job finishing; its two missing
percentages remain dashes until then.

## Remaining-time estimate — approximately 08:40 CDT

Job 451562 has run for about ten minutes. Both generation stages have completed,
and execution-based SQL scoring is underway. It is still running with its 128 GiB
host-memory allocation; no new failure is recorded. The controller heartbeat is
fresh and all three stages in its latest completed cycle returned zero.

Estimated remaining time: **20–45 minutes for evaluation, then 5–10 minutes for
paper refresh and validation** (roughly 25–55 minutes total). This estimate uses
the completed SQL run and the current stage; scoring has no per-example progress
counter and can vary with the generated queries. A further failure would extend
the estimate. The four-hour Slurm limit is a resource ceiling, not the forecast.

The current PDF already contains all six completed campaigns. Only the two
SQL/Llama3B unlikelihood percentages await measurement. The automatic controller
will ingest a successful receipt and rebuild the tables/PDF; the earlier 32 holds
will remain in place.
