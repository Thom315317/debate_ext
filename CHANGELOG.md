# Changelog

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
