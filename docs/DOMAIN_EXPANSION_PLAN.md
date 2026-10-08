# New domains for generality

**Priority decision, 2026-10-03:** the user placed new domains and contrastive baselines
first, before further legacy experiments, full-FT/d2t implementation gaps, and paper
integration. Implement and validate these pilot tracks next. Existing submitted jobs
continue while this work proceeds. See [TASKS.md](../TASKS.md).

Draft plan, 2026-10-02. Requested scope: new domains that test generality.
This is a proposed implementation and experiment sequence, not a preregistration amendment
or a claim that these tasks have been built or run.

## October8 execution update (15:05 CDT)

The three expansion domains are implemented, audited and base-gated under unchanged
criteria. All six original small-model cells and all six explicit-output diagnostics
failed their gates and remain preserved. Larger-model feasibility is registered under
Amendment46, with validity-based seed promotion under Amendment47.

| Expansion task | Current result | Automatic next step |
|---|---|---|
| Unit conversion / Gemma12B | Three complete fresh five-system campaigns at seeds17/42/1234;ordinary SFT improves both directions | Contrastive seed17 training/evaluation, then valid seed42/1234 extension |
| Python↔C++ / Llama8B | Seed17 four-arm training completed;fresh evaluation448717 awaits allocation | Validate paired campaign, then seed42/1234 four-arm replications regardless of effect sign |
| Boolean formula↔truth table | All tested original small/large and explicit-prompt gates fail | Keep failed cells blocked;any genuinely new task variant needs its own declared design and audits |

Amendment48 also extends contrastive comparisons to fmt/de→en at8B/12B and
units/Gemma12B. Three seed17 contrastive packs are running. Missing larger-model
fmt prerequisites449015/449016 are queued;passing cells enter the feeder's PLAN
automatically. All larger-model trainer/batching prerequisites have passed.

Broader Python/C++ functions with bounded conditionals/loops/lists, external natural-code
evaluation, and numerical extrapolation/conversion-composition splits are design proposals
below. They are not implemented or queued in the autonomous pipeline. Their existing
proposal status must not be counted as an additional tested domain or completed experiment.
Current implemented Python/C++ scope remains single-return integer arithmetic with
exhaustive33-input equivalence. TASKS.md is the live completion checklist.

## Recommended order

| Priority | Proposed domain | Directions | What it adds | Main risk |
|---|---|---|---|---|
| 1 | `code_translate_py_cpp` | Python function ↔ C++ function | Executable semantics across programming languages, beyond within-Python obfuscation | Compiler/sandbox failures masquerading as model errors |
| 2 | `logic_truth` | Boolean formula ↔ truth table | Exhaustively checkable logical reasoning; reverse answers need not match one gold formula | Base model may fail the reverse gate; equivalent functions can leak across splits |
| 3 | `units` | Quantity in source unit ↔ quantity in target unit | Numerical reasoning with a reversible map and no pretrained factual-relation dependency | Easy conversions may be at ceiling; rounding can create false failures |

Start with the first two in parallel at the data/scorer stage, then build units. Add no new
language-translation pair in this wave: four MT directions already exist.

## 1. Python ↔ C++ translation

Use a restricted, deterministic function subset: integers, booleans, bounded lists, conditionals,
and bounded loops. Generate Python and C++ from the same typed intermediate representation.
Exclude I/O, imports, pointers, recursion, undefined behavior, implementation-dependent integer
widths, and overflow. This tests translation on the declared subset, not unrestricted software.

Score both directions by compilation/parsing plus behavioral equivalence to the reference on
exhaustive finite inputs wherever tractable. Else use fixed held-out differential test cases and
label the score as test-suite correctness, not proof of equivalence. Never require source-string
match. Set the integer/list bounds and execution budgets before scoring model output.

Split by program family/template and normalized behavior, not generated source spelling.
Audit train/eval overlap with existing code and exec corpora. Report template-held-out results
separately; generated programs alone do not establish performance on natural code.

Optional external evaluation: MultiPL-E has translated Python benchmarks and reference execution
infrastructure. Its project is [nuprl/MultiPL-E](https://github.com/nuprl/MultiPL-E).
Inspect licensing, Python/C++ alignment, executability, and overlap before adoption; keep any
external evaluation problems out of all training data. Do not depend on those benchmarks to
supply the training corpus.

Feasibility check on October 2: `g++` is on PATH on the compute host. A reproducible compiler
version and sandbox still need validation on actual job nodes. Compiler startup time, crashes,
and harness timeouts must be reported separately from semantic wrong answers.

## 2. Boolean formula ↔ truth table

Start with four Boolean variables and an explicitly ordered 16-row truth table. The expression
language permits variables, parentheses, NOT, AND, and OR. Parse with an allowlisted grammar;
do not evaluate generated strings with unrestricted Python eval.

Forward correctness: exact equality of the complete truth vector. Reverse correctness: parse
the proposed formula, evaluate all 16 assignments, and compare its truth vector to the input.
Any equivalent formula is accepted. Formula length is a diagnostic, not the main correctness
criterion. A truth table generally does not identify a unique formula.

Split by the Boolean function (the full truth vector), keeping all equivalent formulas for a
function in one split. Deduplicate before choosing split sizes; report the resulting number of
unique functions rather than repeating formulas to claim a larger corpus. Stratify by formula
size and function properties. Set a length bound that can express every reverse gold function
in the chosen corpus, and audit truncation before training.

Include gold, equivalent alternate formula, malformed expression, echo, contradiction, and
single-bit error scorer fixtures. Report base failures and ceilings even if the domain is not
promoted. A simpler variant would be a separately declared task, not a retrospective replacement
that hides the failed gate.

## 3. Unit conversion

Start with exact rational scale conversions for length, mass, and time; include affine
Celsius/Fahrenheit conversion as a separate reported subtask. Each prompt declares the source
and target unit, so the reverse direction restores a determined physical quantity.

Build pairs from exact rational values. Represent answers in a fixed numeric syntax with unit
labels; compare parsed exact values when possible. If finite decimal output is required, freeze
a rounding rule and tolerance from the declared precision before evaluating models. Reject wrong
units, source echoes, and numeric answers that only happen to round into the tolerance band.

Group splits by canonical physical quantity and conversion family. Keep alternate expressions
of the same quantity together. Hold out magnitude ranges or conversion compositions for a
separate generalization split; do not confuse numerical extrapolation with the main collapse
contrast. Include negatives, zero, fractions, and realistic range checks.

Reserve date normalization for a later variant: explicit Gregorian dates with a year and named
month, with no ambiguous numeric dates or time zones in the first version. Python's
[datetime documentation](https://docs.python.org/3.12/library/datetime.html) provides parsing
and formatting primitives. Dates are lower priority because they overlap existing format tasks.

## Shared pilot and promotion protocol

1. **Declare before measuring.** Write a dated preregistration amendment for each selected domain:
   scope, train/eval grouping, strict scorer, base-gate procedure, planned models/arms/seeds,
   directional prediction or explicit exploratory status, and stopping rule. Predictions cannot
   be chosen after inspecting gate or tuned results.
2. **Build and audit on CPU.** Implement the existing five-function domain contract, config,
   generator, disjointness checks, and gold/echo/empty/garbage fixtures. Run oracle tests on a
   compute node with the actual compiler and executor allocation.
3. **Gate before training.** Initially test Llama 3B and Gemma 4B in both directions on a fixed,
   seeded sample of 200 held-out pairs. Use the existing per-domain base-gate requirements;
   report failed gates and never tune the criterion to obtain a pass. Increase the sample before
   freezing a borderline eligibility decision, with that rule preregistered.
4. **Four-arm pilot.** For passing cells, train `sft`, `rev`, `mix50`, and `replay`, seed 17,
   using the existing recipe and resolved arm-budget rules. Evaluate base plus all four in one
   pass. This tests forward-only loss, reverse learnability, directional rescue, and generic
   replay in six possible domain/model cells: at most **24 trained arms** for the first pilot.
5. **Promote by measurement validity.** Expand every passing, valid pilot cell according to the
   declared rule, including null-collapse cells. Do not select only domains showing collapse.
   Add seeds 42/1234; add the resolvable dose ladder only when the corpus supports it. Add the
   reversed training orientation for asymmetric tasks using the same split and recipe.
6. **Then scale.** Gate the 8B/12B models; add at least two seeds for eligible scale cells.
   Add OLMo where its own base gate passes. Run recovery mechanisms only from checkpoints with
   measured loss, with a suitable model-matched never-had control.

Target up to 6,500 training pairs and at least 1,000 held-out pairs per domain when unique,
leakage-free content supports them. These are targets, not quotas; logic and bounded program
families must be counted after semantic deduplication. If fewer than one effective batch falls
in a dose rung, drop that rung using the existing resolution rule.

## Compute and scheduling

- Keep the existing campaign running while builders and scorer audits proceed.
- Gate first; do not submit full grids for unimplemented or ungated domains.
- Use the verified H100/H200 environment; A30 remains excluded pending its CUDA repair.
- Pack arms by domain/model/seed and chain evaluation; keep the feeder's duplicate and failure
  guards. Use the authorized eight-job QoS within its actual scheduler limits.
- Budget the initial wave as six possible gates plus at most 24 trained arms and six evaluation
  passes. Do not quote a GPU-hour total from unrelated domains. Measure the first completed
  adapter per domain/model and size the remaining jobs with a documented safety margin.
- Record failed gates and harness failures in the checklist. A queued job is not a result.

## Implementation checklist

- [x] Preregister the proposed domains and staged promotion rule.
- [x] Build Python/C++ generator, grouped splits, compiler sandbox, and equivalence scorer.
- [x] Build Boolean parser/generator, function-disjoint splits, and exhaustive verifier.
- [x] Build exact unit generator, grouped splits, and numerical verifier.
- [x] Validate all three scorers on compute-node oracle fixtures.
- [x] Run two-model base gates and record every outcome (all six failed).
- [ ] Run four-arm pilots for all eligible domain/model cells.
- [ ] Add registered seed replication and mirrored orientation for valid passing cells.
- [ ] Integrate all outcomes, including nulls and gate failures, into tables and provenance.

Planning references: [task catalogue](TASK_CATALOGUE.md),
[historical candidate rationale](CANDIDATE_DOMAINS.md), and [current checklist](../TASKS.md).


## Implementation checkpoint — October 3, 2026

Units and four-variable Boolean tasks are built/audited; both tested models failed
their unchanged base gates, and no tuned pilot was admitted. Preserve those failures.
The initial Python/C++ pilot is explicitly narrower than the broad subset above:
single-return integer arithmetic with exhaustive33-input behavior, quadratic train,
piecewise-linear validation and cubic test families. Scorer/compiler/split/budget
audits passed. Gates440727/440728 failed; no original-model cell entered a four-arm pilot.
Existing code/exec sources were all unsupported by this restricted overlap interpreter;
this audit cannot establish unrestricted semantic disjointness. Broader integer/list/
loop programs and external natural-code evaluation remain future work.

## October 7 diagnostic revision

Failure-output inspection found wrong output representations in logic, equations or extra
unit labels in units, and incomplete functions in Python/C++. These are observations from
the truncated format-failure dumps, not an exhaustive content-error taxonomy or proof of cause.
Amendment44 declares separate `logic_explicit`, `units_explicit`, and `py_cpp_explicit`
prompt variants. They retain the original semantic data splits and exact scorers, add explicit
direction-specific output contracts, and preserve all original failures. They are post-diagnostic
exploratory variants, not independent new domains. Each corpus has6500/500/1000 rows and zero
cross-split content/ID overlap; compute-node oracle and realized-budget audits passed.
Jobs447365/447366 gate both models on all1000 held-out pairs per task. The feeder admits
seed17 sft/rev/mix50/replay pilots only if unchanged eligibility thresholds pass. Valid nulls
and adverse effects are retained. Gate failures do not trigger threshold relaxation.

The completed Llama3B explicit gates447365 still fail: units0.8%/1.5%, logic0%/0%,
Python/C++83.8%/89.4% forward/reverse strict accuracy. Python/C++ forward format
failure16.0% exceeds the unchanged15% cap; missing closing braces and unsupported
range guards appear in its dumps. Gemma4B explicit gates447366 also fail: units strict
39.9%/26.9%, with forward format failure15.1% exceeding the fixed15% cap;
logic0%/0.1%; Python/C++29.9%/88.8%, with70.1% forward format failures.
These reports are dated October8 UTC (October7 local time). No explicit-format pilot is
admitted; preserve these outcomes without threshold or extraction changes.

Amendment46 advances the planned larger-model feasibility checks: original units,
logic and Python/C++ tasks on Llama3.1-8B and Gemma3-12B, all1000 pairs, unchanged
instructions/scorers/thresholds. The runner admits each passing cell into a seed17
four-arm pilot under `generality_scale_pilot`. Report every failed gate and valid
null/adverse pilot; model scaling does not add independent domains.
