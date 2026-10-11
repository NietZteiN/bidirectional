# Results progress — October 11, 2026 morning CDT

All seven prioritized table-completion campaigns are finished, filling all twelve
missing arm/task/model combinations. There are zero pending campaign arms and zero
missing numeric table fields. Final SQL/Llama3B evaluation 451562 completed at
09:19 CDT with exit code zero, in 48 minutes 34 seconds. Its successful retry used
128 GiB of host memory after the earlier 64 GiB failure; no scientific setting changed.
The paper has refreshed with its results. The 32 earlier unrelated jobs remain held.

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

The final SQL/Llama3B comparison is:

| Method | Forward success (%) | Backward success (%) |
|---|---:|---:|
| Base | 49.1 | 51.4 |
| Forward SFT | 62.0 | 24.1 |
| 5% reversal | 60.8 | 55.4 |
| Unlikelihood | 62.1 | 3.6 |
| Round-trip | 61.9 | 48.9 |

Here, 5% reversal preserves backward success with a small forward cost relative to
SFT. Unlikelihood has the lowest backward success; that negative result is retained.

## Newly completed small-corpus comparisons

Relations: forward/backward success is 86.7%/71.1% for SFT, 88.1%/73.3% for
1% reversal and 87.4%/75.6% for 5%. Execution prediction: 29.0%/13.0% for SFT,
27.0%/14.0% for 1% and 30.5%/17.5% for 5%. These exploratory small-corpus
experiments show modest gains; they do not reproduce the dramatic directional
loss in formatting or code.

The refreshed audit independently replays stored binary scores and finds no
mismatches in 564 printed table percentages or 333 registered numerical values.
It also verifies 762 source hashes, 21 corpus split sizes and 20 frozen translation
threshold rows, replaying 125 distinct stored campaigns. The final
`TABLE_COMPLETION_PREFLIGHT_2026-10-11.json` binds these checks to the completed
document/report hashes. Strict PDF checks pass: seven body pages, 26 total pages,
126-word abstract, embedded fonts and resolved references/numerical macros.

All five detached-controller health checks pass. No GPU experiment is currently
running; the CPU controller remains healthy and idle, with the earlier holds intact.
The full regression suite passed 441 tests after the manuscript-folder move.

## Earlier remaining-time estimate — approximately 08:40 CDT (closed)

Job 451562 has run for about ten minutes. Both generation stages have completed,
and execution-based SQL scoring is underway. It is still running with its 128 GiB
host-memory allocation; no new failure is recorded. The controller heartbeat is
fresh and all three stages in its latest completed cycle returned zero.

Estimated remaining time: **20–45 minutes for evaluation, then 5–10 minutes for
paper refresh and validation** (roughly 25–55 minutes total). This estimate uses
the completed SQL run and the current stage; scoring has no per-example progress
counter and can vary with the generated queries. A further failure would extend
the estimate. The four-hour Slurm limit is a resource ceiling, not the forecast.

That forecast is now closed: evaluation completed at 09:19 CDT and the controller
incorporated the final two SQL percentages into the tables/PDF. The earlier 32 holds
remain in place; their experiments were outside this completion forecast.
