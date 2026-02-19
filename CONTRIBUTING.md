# Contributing

Thank you for your interest in debate_ext!

## Development Setup

1. Clone the repository
2. Run `cd vscode-extension && npm install`
3. Run `npm run compile`
4. Press F5 in VS Code to launch the Extension Development Host

## Running Tests

```bash
cd vscode-extension
npm run compile
npm run smoke    # build + security checks
npm test         # unit tests
```

## Benchmark Development

See the [README](README.md) for benchmark commands.
Use `--dry-run --limit 5` for testing without API calls.

## Pull Requests

- One feature per PR
- Include test coverage for new functions
- Run `npm run smoke` before submitting
