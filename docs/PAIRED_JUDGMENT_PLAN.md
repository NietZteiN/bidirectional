# Cross-domain paired judgments — October 11, 2026

The user chooses a separately labeled paired-judgment baseline for translation and SQL.
Original code-equivalence CFT remains code-specific. Its n/a cells mean not applicable;
the new PJ-5 comparisons will appear separately after measurement.

- Four cells: English→German and question→SQL, Llama3B and Llama8B, seed17.
- Two new adapters per cell: PJ-5 and a shuffled-label control.
- PJ-5 uses the same selected pair IDs as 5% reversal, replacing generation with nearly
  balanced Yes/No judgments. All remaining rows are forward generation. There is no
  backward generation CE. Example counts and optimizer steps match the saved SFT runs;
  tokens/runtime are measured separately.
- Translation has 6500 rows; SQL has 6499. Each replaces 325 rows with 162 positive
  and 163 negative judgments. All negative candidates come from training data. SQL
  candidates stay in the same database and must execute and fail the frozen equivalence
  check; errors cannot manufacture negatives. Unrecorded translation paraphrases remain
  a potential source of negative-label noise.
- Shuffled labels retain identical inputs and class counts. Compare PJ-5 against
  SFT, 5% reversal and that control in both original generation tasks.
- Four-step training checks explicitly consume 256 rows (224 forward,16 Yes,16 No).
  After both new arms pass, train full adapters, check shared generation on40 pairs,
  then evaluate the complete held-out test split with a fresh base/SFT/mix5 comparison.
- At most one phase per cell, three bounded attempts and no automatic retry of a
  deterministic failure. SQL evaluation reserves192 GiB host memory; other phases64 GiB.
- New sources/outputs use an isolated namespace and preserve all existing worker signatures.
  The detached controller runs `125_paired_judgment_queue.py`; paper refresh runs
  `126_paired_judgment_report.py`. Completed rates appear in the main paper and paired
  intervals in the appendix, including adverse outcomes. Pending runs never supply values.

Live queue: `runs/feeder/paired_judgment_status.json`. Configuration:
`configs/paired_judgment.json`. Methods: `paper/bidir/planned_evaluations.tex`.
Source audit: `paper/bidir/PAIRED_JUDGMENT_RESULTS.json`.

Run `python scripts/127_paired_judgment_preflight.py` in the existing CPU allocation
before admitting jobs. It validates baseline adapter bytes and token filtering for all
eight new arms. The scheduler rejects missing, incomplete or changed preflight records.
The GPU checks remain mandatory after this CPU/data check.

October11 update: all four GPU checks (452104–452107) passed. Amendment60 gives
this campaign exclusive project GPU priority through full evaluation. Twenty-five other
GPU jobs are held; two running training packs were requeued. The five-second guard in
`scripts/128_paired_judgment_priority.py` persists after logout and CPU-host renewal,
keeps phase gaps exclusive, and releases only its own recorded holds when all four cells
complete. Broad scheduling is disabled in the meantime. Scientific worker signatures,
the paired-data plan and original results are unchanged.

October12 completion: all four cells passed their four phases on the first attempt.
The last full evaluation finished at 00:12 CDT; the priority guard released only its
25 recorded holds at 00:13 CDT. Both-direction rates and paired intervals are in
`paper/bidir/tables/paired_judgment_results.tex`; intervals are in
`paper/bidir/tables/paired_judgment_intervals.tex`. Five-percent reversed-pair SFT has
higher backward success than PJ-5 in three cells; their translation/Llama3B difference
is unresolved. PJ-5 versus shuffled labels is positive on SQL/Llama3B, negative on
translation/Llama8B, and unresolved in the remaining two cells. These results do not
support a general advantage from the judgment labels. All adverse comparisons remain
reported; no extra seed or modified objective has been selected from these outcomes.
