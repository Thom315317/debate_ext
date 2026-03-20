# Changelog

## [2.0.0] — 2026-03-19

### Added
- `--judge-strategy scoring|binary` flag: scoring (default, existing A) or binary (new B — one pass/fail prediction per candidate)
- `--no-judge` flag: skip judge evaluation entirely (generation + exec only)
- Binary strategy B: each judge independently predicts pass/fail with confidence per candidate, tiebreaker on divergence (opposite verdict or |Δconfidence| > 0.3)

### Fixed
- **CRITICAL**: Claude judge temperature was absent (API default 1.0). Now explicitly set to 0.2 like GPT and Gemini.
- `--help` now shows 14 configs (was 8)
- `--merge` and `--rejudge-from` now use dataset from source file instead of hardcoded MBPP+
- THIRD_PARTY_NOTICES: added HumanEval+ / EvalPlus attribution
- README: updated to v2.0.0, added HumanEvalPlus.jsonl to data/ listing
- vscode-extension/README: Node.js v18 (was v16)

### Changed
- All judges (Claude, GPT, Gemini) now use temperature=0.2 uniformly
- `judgeStrategy` and `noJudge` saved in report metadata

## [1.1.1] — 2026-03-15

### Fixed
- `extractPythonCode()` fallback for models without markdown fences (MiniMax M2, GLM-4.6)
  - When no ` ```python ``` ` block is found, now extracts `def` functions + `import`/`from` from raw text
  - Fixes 98/164 false-empty selfrefine results (MiniMax) and 104/164 (GLM-4.6)
- SelfRefine/XFusion no longer overwrites valid code with empty string
  - When review/fusion extraction fails, previous code is kept instead of being replaced by `""`
  - Remaining empty codes dropped from ~15 to 0 (GLM selfrefine)
- Circuit-breaker no longer treats "all configs empty" as a judge failure
  - Tasks where all configs produce no code are skipped and retried on `--resume`
  - Previously these tasks triggered the circuit-breaker exit, causing premature benchmark abort

## [1.1.0] — 2026-02-23

### Added
- 3 cross-review collaboration modes (ximprove, xbest, xfusion) — 6 new configs (14 total)
  - ximprove: both generate, other reviews, primary improves
  - xbest: both generate & cross-review, best selected by test execution
  - xfusion: both generate & cross-review, primary fuses the best of both
- HumanEval+ dataset support (`--dataset humaneval`, 164 tasks)
- `setup_humanevalplus.py` script for data preparation
- ROADMAP.md with planned improvements

### Changed
- Default gen2 model: DeepSeek-V3.1 671B (was MiniMax M2)
- Config count: 14 (was 8)
- Code truncation limit for judges: 8000 chars (was 4000)

### Fixed
- Judge labels limited to A–H (8): configs 9–14 were never scored. Now uses A–Z.
- Function renaming in all generation/fix/fusion prompts: `entry_point` now enforced, eliminating NameError failures (33–73% of prior errors)
- Code dedup before judge evaluation: identical codes sent once, scores propagated to all matching configs

## [1.0.0] — 2025-02-19

### Added
- Paper-grade benchmark suite (`benchmark_paper.ts`)
  - 8 collaboration configurations (solo, lead, orch, selfrefine × 2 generators)
  - 3-level judge evaluation (R1 → R2 debate → Tie-breaker)
  - Blind judge mode (no exec status leakage)
  - EvalPlus+ extended test evaluation
  - Statistical metrics: pass@k, McNemar, AUC, Spearman, Cohen's κ
- Workflow features: --rejudge-from, --merge, --offset, --resume
- Circuit-breaker on judge API failures
- Smoke test suite
- MBPP+ data loader (378 tasks)

### VS Code Extension
- Multi-agent debate sidebar (Claude + GPT)
- Auto-detect task complexity
- Streaming output with think-block detection
