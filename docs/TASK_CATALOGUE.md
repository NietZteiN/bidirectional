# Bidirectional Reasoning Tasks Beyond Deobfuscation

> **Provenance.** Written by the project's authors as a candidate menu, added to the repository
> 2026-09-10. It is a *menu*, not a plan: the domains this paper actually runs are fixed in
> [`../RUN_PLAN.md`](../RUN_PLAN.md), and which of these are worth adding — with what each buys
> and what it costs — is argued in [`CANDIDATE_DOMAINS.md`](CANDIDATE_DOMAINS.md).
>
> The **Key references** section at the end was truncated in the source and is left as it
> arrived rather than reconstructed from memory; citations for anything promoted out of this
> file must be verified before use.

Candidate tasks for testing whether LLMs can reason in both directions of a transformation, and whether reversed-pair training data closes the gap between directions. Each task is sized for a team of two.

## Legend

- `L` **Lossy.** The forward map is many-to-one, so the backward direction recovers discarded information. This is the closest analog to deobfuscation.
- `S` **Semantic.** Both directions are one-to-many, and correctness means semantic equivalence rather than string match.
- `B` **Bijective.** An exact or near-exact inverse exists. These make good low-difficulty controls.

## Shared protocol for every team

- [ ] Write a programmatic generator for the easy direction so paired data is unlimited
- [ ] Write an automatic verifier for the hard direction so grading is free and exact
- [ ] Use the same model family and token budget as every other team
- [ ] Train three arms (forward only, reversed pairs only, and both)
- [ ] Report forward accuracy, backward accuracy, round-trip accuracy, and the asymmetry gap

---

## 1. Lossy code transforms (closest analogs to deobfuscation)

- [ ] **1. Compilation ↔ decompilation** `L`
  - **Forward** C source → x86 or ARM assembly
  - **Backward** Assembly → C source
  - **Verifier and data** Recompile and run tests; ExeBench, Decompile-Eval (LLM4Decompile)
  - **Team** ______ , ______

- [ ] **2. Source ↔ IR or bytecode** `L`
  - **Forward** C → LLVM IR, Java → JVM bytecode, Python → `dis` output, C or Rust → WebAssembly text
  - **Backward** IR or bytecode → source
  - **Verifier and data** Reassemble and run differential tests; generate with clang, javac, dis
  - **Team** ______ , ______

- [ ] **3. Optimization level** `L`
  - **Forward** -O0 code → -O3 code
  - **Backward** Optimized code → unoptimized, readable code
  - **Verifier and data** Differential testing on random inputs
  - **Team** ______ , ______

- [ ] **4. Minification ↔ beautification** `L`
  - **Forward** Readable JavaScript → minified JavaScript
  - **Backward** Minified JavaScript → readable code with recovered identifier names
  - **Verifier and data** Tests plus name-recovery accuracy; terser as generator
  - **Team** ______ , ______

- [ ] **5. Transpilation** `L`
  - **Forward** TypeScript → JavaScript, ES2022 → ES5 via Babel
  - **Backward** Transpiled output → original source
  - **Verifier and data** Type-check and tests
  - **Team** ______ , ______

- [ ] **6. Type erasure ↔ type inference** `L`
  - **Forward** Strip type annotations (Python, TypeScript)
  - **Backward** Restore type annotations
  - **Verifier and data** mypy or tsc agreement, exact match; ManyTypes4Py
  - **Team** ______ , ______

- [ ] **7. Macro expansion ↔ macro recovery** `L`
  - **Forward** Expand C preprocessor or Rust macros
  - **Backward** Re-abstract expanded code into macros
  - **Verifier and data** Re-expand and compare
  - **Team** ______ , ______

- [ ] **8. Inlining ↔ function extraction** `L`
  - **Forward** Inline helper functions
  - **Backward** Re-extract helper functions
  - **Verifier and data** Tests plus structural match
  - **Team** ______ , ______

- [ ] **9. Desugaring ↔ resugaring** `L`
  - **Forward** Comprehensions, decorators, `with` blocks, f-strings → primitive forms
  - **Backward** Primitive forms → idiomatic sugar
  - **Verifier and data** Tests; AST rewriters as generators
  - **Team** ______ , ______

- [ ] **10. Loop unrolling ↔ rerolling** `L`
  - **Forward** Unroll a loop k times
  - **Backward** Recover the original loop
  - **Verifier and data** Tests
  - **Team** ______ , ______

- [ ] **11. SSA ↔ source** `L`
  - **Forward** Lower code to three-address or SSA form
  - **Backward** Raise SSA form back to structured source
  - **Verifier and data** Tests
  - **Team** ______ , ______

- [ ] **12. Comment stripping ↔ restoration** `L`
  - **Forward** Remove comments and docstrings
  - **Backward** Regenerate comments and docstrings
  - **Verifier and data** Weak verifier (reference similarity only), secondary task
  - **Team** ______ , ______

---

## 2. Code translation and migration

- [ ] **13. Cross-language translation** `S`
  - **Forward** Python → C++, Java → C#, Python → Rust, C → Go
  - **Backward** The reverse of each pair
  - **Verifier and data** Unit tests; TransCoder test set, AVATAR, CodeTransOcean, xCodeEval, HumanEval-X, MultiPL-E
  - **Team** ______ , ______

- [ ] **14. Paradigm translation** `S`
  - **Forward** Loops → map/filter/reduce, recursion → iteration, OOP → procedural
  - **Backward** The reverse of each
  - **Verifier and data** Tests
  - **Team** ______ , ______

- [ ] **15. Concurrency style** `S`
  - **Forward** Callbacks → promises → async/await, threads → asyncio
  - **Backward** The reverse of each
  - **Verifier and data** Tests with deterministic scheduling
  - **Team** ______ , ______

- [ ] **16. Version migration** `S`
  - **Forward** Python 2 → 3, Java 8 → 17, COBOL → Java
  - **Backward** The reverse of each
  - **Verifier and data** Tests; 2to3 as a one-way oracle
  - **Team** ______ , ______

- [ ] **17. Library migration** `S`
  - **Forward** NumPy → JAX, pandas → polars, PyTorch → TensorFlow
  - **Backward** The reverse of each
  - **Verifier and data** Numerical equivalence on random inputs
  - **Team** ______ , ______

- [ ] **18. Scalar ↔ vectorized** `S`
  - **Forward** Python loops → NumPy, serial → OpenMP, CPU → CUDA
  - **Backward** The reverse of each
  - **Verifier and data** Output equality plus speedup
  - **Team** ______ , ______

- [ ] **19. SQL dialects** `S`
  - **Forward** PostgreSQL → MySQL → SQLite → Spark SQL
  - **Backward** The reverse of each
  - **Verifier and data** Execute on the same database; sqlglot as oracle
  - **Team** ______ , ______

- [ ] **20. Build and infrastructure configs** `S`
  - **Forward** Makefile → CMake, Dockerfile → docker-compose, Terraform → CloudFormation
  - **Backward** The reverse of each
  - **Verifier and data** Build success, plan diff
  - **Team** ______ , ______

- [ ] **21. Hardware description** `S` / `L`
  - **Forward** Verilog → VHDL, RTL → gate-level netlist
  - **Backward** The reverse of each
  - **Verifier and data** Simulation equivalence; VerilogEval
  - **Team** ______ , ______

---

## 3. Code and execution (forward simulation vs inverse inference)

- [ ] **22. Output ↔ input prediction** `L`
  - **Forward** Program + input → output
  - **Backward** Program + output → an input that produces it
  - **Verifier and data** Execute; CRUXEval, CodeI/O
  - **Team** ______ , ______

- [ ] **23. Program ↔ execution trace** `L`
  - **Forward** Program → line-by-line trace
  - **Backward** Trace → program
  - **Verifier and data** Re-trace and compare
  - **Team** ______ , ______

- [ ] **24. Program ↔ I/O examples** `L`
  - **Forward** Program → input/output examples
  - **Backward** Examples → program (programming by example)
  - **Verifier and data** Held-out examples; FlashFill-style benchmarks
  - **Team** ______ , ______

- [ ] **25. Pre ↔ postcondition** `S`
  - **Forward** Code + precondition → strongest postcondition
  - **Backward** Code + postcondition → weakest precondition
  - **Verifier and data** Z3, Dafny
  - **Team** ______ , ______

- [ ] **26. Spec ↔ code** `S`
  - **Forward** Code → formal specification (Dafny, JML)
  - **Backward** Specification → code
  - **Verifier and data** Verifier passes
  - **Team** ______ , ______

- [ ] **27. Code ↔ tests** `S`
  - **Forward** Code → test suite
  - **Backward** Test suite → code
  - **Verifier and data** Mutation score, pass rate
  - **Team** ______ , ______

- [ ] **28. Bug injection ↔ repair** `L`
  - **Forward** Correct code → mutated buggy code
  - **Backward** Buggy code → fixed code
  - **Verifier and data** Tests; Defects4J, BugsInPy, mutation tools
  - **Team** ______ , ______

- [ ] **29. Patch algebra** `B`
  - **Forward** Before + patch → after
  - **Backward** Before + after → patch, and after + patch → before
  - **Verifier and data** `git apply` exact match
  - **Team** ______ , ______

- [ ] **30. Cellular automata** `L`
  - **Forward** Game of Life state → next state
  - **Backward** State → a valid predecessor
  - **Verifier and data** Simulate; backward is NP-hard in general, a useful hard ceiling
  - **Team** ______ , ______

- [ ] **31. Chess notation** `L`
  - **Forward** Move list (PGN) → board position (FEN)
  - **Backward** Board position → a legal move sequence reaching it
  - **Verifier and data** python-chess legality check
  - **Team** ______ , ______

---

## 4. Code and other representations

- [ ] **32. Code ↔ natural language** `S`
  - **Forward** Code → summary
  - **Backward** Description → code
  - **Verifier and data** Round-trip correctness (Allamanis et al., 2024); CodeSearchNet, CoNaLa
  - **Team** ______ , ______

- [ ] **33. Code ↔ pseudocode** `S`
  - **Forward** C++ → line-level pseudocode
  - **Backward** Pseudocode → C++
  - **Verifier and data** Tests; SPoC
  - **Team** ______ , ______

- [ ] **34. Code ↔ AST or CFG** `B` (AST) / `L` (CFG)
  - **Forward** Code → serialized AST or control-flow graph
  - **Backward** AST or CFG → code
  - **Verifier and data** Parser round trip
  - **Team** ______ , ______

- [ ] **35. Code ↔ diagrams** `S`
  - **Forward** Classes → PlantUML or Mermaid UML, ORM models → SQL DDL
  - **Backward** The reverse of each
  - **Verifier and data** Structural comparison after parsing
  - **Team** ______ , ______

- [ ] **36. Regex ↔ description or strings** `S`
  - **Forward** Regex → natural-language description or matching strings
  - **Backward** Description or strings → regex
  - **Verifier and data** DFA equivalence; NL-RX, KB13, StructuredRegex
  - **Team** ______ , ______

- [ ] **37. Grammar ↔ strings** `L`
  - **Forward** Context-free grammar → sample strings
  - **Backward** Strings → grammar
  - **Verifier and data** Membership testing
  - **Team** ______ , ______

- [ ] **38. SQL ↔ natural language** `S`
  - **Forward** SQL → explanation
  - **Backward** Question → SQL
  - **Verifier and data** Execution accuracy; Spider, BIRD
  - **Team** ______ , ______

- [ ] **39. Query ↔ result** `L`
  - **Forward** SQL + database → result table
  - **Backward** Database + result table → a query that produces it
  - **Verifier and data** Execute
  - **Team** ______ , ______

- [ ] **40. SQL ↔ dataframe code** `S`
  - **Forward** SQL → pandas → relational algebra
  - **Backward** The reverse of each
  - **Verifier and data** Execute on the same data
  - **Team** ______ , ______

- [ ] **41. Shell ↔ natural language** `S`
  - **Forward** Bash command → description
  - **Backward** Description → bash command
  - **Verifier and data** Sandboxed execution; NL2Bash, InterCode
  - **Team** ______ , ______

- [ ] **42. Diff ↔ commit message** `S`
  - **Forward** Diff → commit message
  - **Backward** Commit message + old code → diff
  - **Verifier and data** Tests on the generated diff
  - **Team** ______ , ______

- [ ] **43. API spec ↔ client code** `S`
  - **Forward** OpenAPI or protobuf spec → client code
  - **Backward** Client code → spec
  - **Verifier and data** Schema validation, compile
  - **Team** ______ , ______

---

## 5. Structured data and encodings (mostly controls)

- [ ] **44. JSON ↔ XML ↔ YAML ↔ TOML** `B` with lossy edges
  - **Forward / backward** All pairs; lossy edges include XML attributes, YAML anchors and comments, TOML nesting limits
  - **Verifier and data** Parse and compare trees
  - **Team** ______ , ______

- [ ] **45. Table formats** `B`
  - **Forward / backward** CSV ↔ JSON ↔ Markdown ↔ HTML tables
  - **Verifier and data** Parse and compare
  - **Team** ______ , ______

- [ ] **46. Document formats** `B` / `L`
  - **Forward / backward** Markdown ↔ HTML ↔ LaTeX
  - **Verifier and data** Pandoc as oracle
  - **Team** ______ , ______

- [ ] **47. Nested ↔ flat data** `L`
  - **Forward** Flatten JSON, denormalize relational tables
  - **Backward** Re-nest, normalize
  - **Verifier and data** Re-nest and compare
  - **Team** ______ , ______

- [ ] **48. Instance ↔ schema** `L`
  - **Forward** JSON instances → JSON Schema
  - **Backward** JSON Schema → valid instances
  - **Verifier and data** jsonschema validation
  - **Team** ______ , ______

- [ ] **49. Graph formats** `B`
  - **Forward / backward** Edge list ↔ adjacency matrix ↔ adjacency list ↔ DOT
  - **Verifier and data** Graph comparison
  - **Team** ______ , ______

- [ ] **50. Byte encodings** `B`
  - **Forward / backward** Encode and decode base64, hex, URL encoding, Unicode escapes
  - **Verifier and data** Exact match
  - **Team** ______ , ______

- [ ] **51. Classical ciphers** `B` with key, `L` without
  - **Forward** Encrypt with Caesar/ROT-k, Vigenère, substitution
  - **Backward** Decrypt with and without the key
  - **Verifier and data** Exact match; McCoy et al. (2023) as baseline
  - **Team** ______ , ______

- [ ] **52. Number and unit systems** `B`
  - **Forward / backward** Decimal ↔ binary ↔ hex, Roman numerals, unit conversions, date formats and time zones
  - **Verifier and data** Exact match
  - **Team** ______ , ______

- [ ] **53. Compression** `B`
  - **Forward / backward** Run-length, LZ with explicit dictionary, Huffman with given table
  - **Verifier and data** Exact match
  - **Team** ______ , ______

- [ ] **54. Binary layouts** `B`
  - **Forward / backward** C struct ↔ byte layout with padding and endianness
  - **Verifier and data** Deserialize and compare
  - **Team** ______ , ______

---

## 6. Math and formal reasoning

- [ ] **55. Differentiation ↔ integration** `L`
  - **Forward** Function → derivative
  - **Backward** Derivative → antiderivative
  - **Verifier and data** SymPy; Lample and Charton (2020)
  - **Team** ______ , ______

- [ ] **56. Expansion ↔ factorization** `L`
  - **Forward** Expand polynomials, multiply integers
  - **Backward** Factor polynomials and integers
  - **Verifier and data** SymPy
  - **Team** ______ , ______

- [ ] **57. Forward ↔ backward word problems** `S`
  - **Forward** Solve the problem
  - **Backward** Given the answer, infer a masked quantity in the question
  - **Verifier and data** Exact match; FOBAR (Jiang et al., 2024), RevThink (Chen et al., 2025)
  - **Team** ______ , ______

- [ ] **58. Autoformalization ↔ informalization** `S`
  - **Forward** Natural-language math → Lean 4 or Isabelle
  - **Backward** Formal statement → natural language
  - **Verifier and data** Lean type-check; miniF2F, ProofNet
  - **Team** ______ , ______

- [ ] **59. Formula ↔ truth table** `L`
  - **Forward** Boolean formula → truth table
  - **Backward** Truth table → formula
  - **Verifier and data** Evaluation
  - **Team** ______ , ______

- [ ] **60. Normal forms** `L`
  - **Forward** Formula → CNF or DNF
  - **Backward** CNF or DNF → compact formula
  - **Verifier and data** SAT equivalence
  - **Team** ______ , ______

- [ ] **61. Regex ↔ DFA** `S`
  - **Forward** Regex → DFA (Thompson and subset construction)
  - **Backward** DFA → regex (state elimination)
  - **Verifier and data** Equivalence check
  - **Team** ______ , ______

- [ ] **62. Deduction ↔ abduction** `L`
  - **Forward** Rules + facts → conclusion
  - **Backward** Rules + conclusion → missing fact
  - **Verifier and data** Symbolic checker; ProofWriter, RuleTaker
  - **Team** ______ , ______

- [ ] **63. Expression notation** `B`
  - **Forward / backward** Infix ↔ postfix ↔ prefix ↔ LaTeX ↔ SymPy code
  - **Verifier and data** Evaluate
  - **Team** ______ , ______

---

## 7. Natural language and knowledge

- [ ] **64. Fact reversal** `B`
  - **Forward** "A is B", child → parent
  - **Backward** "B is A", parent → child
  - **Verifier and data** Exact match; Berglund et al. (2023), Golovneva et al. (2024)
  - **Team** ______ , ______

- [ ] **65. Machine translation** `S`
  - **Forward** Language X → Y
  - **Backward** Language Y → X, comparing high- and low-resource directions
  - **Verifier and data** chrF, COMET; FLORES-200
  - **Team** ______ , ______

- [ ] **66. Diacritic restoration** `L`
  - **Forward** Strip diacritics (trivial)
  - **Backward** Restore diacritics (Vietnamese, French, Yoruba)
  - **Verifier and data** Exact match; unlimited data
  - **Team** ______ , ______

- [ ] **67. Script conversion** `L` one way
  - **Forward / backward** Kanji and kana ↔ romaji, hanzi ↔ pinyin, Traditional ↔ Simplified Chinese
  - **Verifier and data** Exact match
  - **Team** ______ , ______

- [ ] **68. Grapheme ↔ phoneme** `S`
  - **Forward** Spelling → IPA
  - **Backward** IPA → spelling
  - **Verifier and data** Phoneme error rate; CMUdict
  - **Team** ______ , ______

- [ ] **69. Spelling and string reversal** `B`
  - **Forward** Word → letters, string → reversed string
  - **Backward** Letters → word, reversed string → original
  - **Verifier and data** Exact match; probes tokenization blind spots
  - **Team** ______ , ______

- [ ] **70. Style transfer** `S`
  - **Forward / backward** Formal ↔ informal, active ↔ passive, complex ↔ simple
  - **Verifier and data** Style classifier plus content preservation; GYAFC
  - **Team** ______ , ______

- [ ] **71. Parsing ↔ generation** `S`
  - **Forward** Sentence → AMR, dependency tree, or logical form
  - **Backward** Structure → sentence
  - **Verifier and data** Smatch; AMR 3.0
  - **Team** ______ , ______

- [ ] **72. Summarization ↔ expansion** `L`
  - **Forward** Text → summary
  - **Backward** Summary → full text
  - **Verifier and data** Weak verifier, secondary task
  - **Team** ______ , ______

- [ ] **73. Question ↔ answer** `S`
  - **Forward** Question → answer
  - **Backward** Answer → question (Jeopardy style)
  - **Verifier and data** Weak verifier, secondary task
  - **Team** ______ , ______

---

## 8. Science and other domains

- [ ] **74. SMILES ↔ IUPAC name** `B`
  - **Forward / backward** Molecular string ↔ systematic name
  - **Verifier and data** RDKit canonicalization; PubChem
  - **Team** ______ , ______

- [ ] **75. Molecule ↔ description** `S`
  - **Forward** Molecule → caption
  - **Backward** Caption → molecule
  - **Verifier and data** ChEBI-20, MolT5
  - **Team** ______ , ______

- [ ] **76. Reaction ↔ retrosynthesis** `L`
  - **Forward** Reactants → product
  - **Backward** Product → reactants
  - **Verifier and data** Top-k exact match; USPTO-50k
  - **Team** ______ , ______

- [ ] **77. DNA ↔ protein** `L`
  - **Forward** Codon translation (many-to-one)
  - **Backward** Back-translation (one-to-many)
  - **Verifier and data** Exact forward, codon-usage plausibility backward
  - **Team** ______ , ______

- [ ] **78. Music notation** `B` / `S`
  - **Forward / backward** ABC notation ↔ MIDI events, chords ↔ Roman numeral analysis
  - **Verifier and data** music21
  - **Team** ______ , ______

- [ ] **79. Digital circuits** `L`
  - **Forward / backward** Boolean expression ↔ gate netlist ↔ truth table
  - **Verifier and data** Simulation
  - **Team** ______ , ______

- [ ] **80. Geometry and location** `B` / `S`
  - **Forward / backward** Coordinates ↔ geohash, SVG path ↔ shape description
  - **Verifier and data** Exact match or render and compare
  - **Team** ______ , ______

---

## Suggested core slate

A balanced set that spreads teams across the three labels and across code and non-code tasks.

- [ ] **#1 Compilation ↔ decompilation** (`L`, code)
- [ ] **#4 Minification ↔ beautification** (`L`, code)
- [ ] **#22 Output ↔ input prediction** (`L`, execution)
- [ ] **#13 Cross-language translation** (`S`, code)
- [ ] **#55 Differentiation ↔ integration** (`L`, non-code)
- [ ] **#66 Diacritic restoration** (`L`, non-code)
- [ ] **#44 JSON ↔ XML ↔ YAML** (`B`, control where a reversed-pair effect should be small)

Tasks with only weak verifiers (#12, #72, #73) are better kept out of the core slate, since noisy grading would blur the effect being measured.

---

## Key references

- Allamanis, Panthaplackel, and

> *(truncated in the source document)*
