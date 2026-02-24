# Roadmap

Planned and completed improvements for debate_ext.

## Done (v1.1)

- **Cross-review modes** (ximprove, xbest, xfusion): both models generate independently, then cross-review and improve. Three variants: primary-improved, best-by-test-execution, and fusion.
- **HumanEval+ dataset**: 164 tasks from OpenAI HumanEval, augmented by EvalPlus. Use `--dataset humaneval`.
- **DeepSeek-V3.1 as gen2**: replaced MiniMax M2 with DeepSeek-V3.1 671B for a quasi-symmetric pair.
- **Multi-pass support**: `--max-iter N` controls review iterations for all collaboration modes.
- **Judge evaluation fixes**: labels extended to A–Z (was A–H, blocking 6 configs), code dedup with score propagation, `entry_point` enforced in all prompts to prevent function renaming.

## Planned

### Collaboration Modes

- **Error-feedback repair**: generate code, execute tests, feed error output back to the model, regenerate. Currently selfrefine reviews code "blind" without test results.
- **Confidence gating**: if generated code already passes base tests, skip refinement.

### Execution Safety

- **Docker sandbox** (`--sandbox docker`): run generated code in an isolated container instead of bare `python3 -c`.

### Dataset Support

- **Additional benchmarks**: APPS, CodeContests, LiveCodeBench.

### Model Coverage

- **More generator pairs**: Llama 4, CodeGemma — test various symmetric pairs.
- **Local generation**: support local Ollama models alongside cloud for cost-free generation.

### Analysis

- **Interactive dashboard**: HTML report with per-task drill-down, judge agreement heatmaps.
- **Automatic paper tables**: LaTeX/CSV export of benchmark results.
