# Roadmap

Planned improvements for future versions of debate_ext.

## Collaboration Modes

- **Error-feedback repair**: generate code, execute tests, feed error output back to the model, regenerate. Currently selfrefine reviews code "blind" without test results.
- **Cross-review mode**: gen1 generates, gen2 reviews (and vice versa). Combines the strengths of lead and selfrefine.
- **Confidence gating**: if generated code already passes base tests, skip refinement. Prevents degradation observed with gen2-selfrefine.

## Execution Safety

- **Docker sandbox** (`--sandbox docker`): run generated code in an isolated container instead of bare `python3 -c`.

## Dataset Support

- **HumanEval+**: 164 tasks from OpenAI HumanEval, augmented by EvalPlus. (v1.1)
- **Additional benchmarks**: APPS, CodeContests, LiveCodeBench.

## Model Coverage

- **More generator pairs**: DeepSeek-V3, Llama 4, CodeGemma — test symmetric (strong+strong) collaboration.
- **Local generation**: support local Ollama models alongside cloud for cost-free generation.

## Analysis

- **Interactive dashboard**: HTML report with per-task drill-down, judge agreement heatmaps.
- **Automatic paper tables**: LaTeX/CSV export of benchmark results.
