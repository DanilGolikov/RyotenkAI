# Contributing to RyotenkAI

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/ryotenkai.git`
3. Run setup: `bash setup.sh`
4. Create a feature branch: `git checkout -b feature/my-feature`

## Development Workflow

### Code Style

- **Formatter**: Black (line-length 120)
- **Linter**: Ruff with strict rules
- **Type checking**: MyPy + Pyright (gradual adoption)
- **Pre-commit hooks** enforce all of the above on every commit

### Running Tests

```bash
make test          # all tests
make test-unit     # unit tests only
make test-fast     # skip slow/integration tests
make test-cov      # with coverage report
```

### Running Linters

```bash
make lint          # check only
make format        # auto-format
make fix-all       # auto-fix all fixable issues
```

## Pull Request Process

1. Ensure your code passes all tests and linters
2. Update documentation if you changed public APIs
3. Write a clear PR description explaining **what** and **why**
4. Keep PRs focused — one feature or fix per PR

## Code Architecture

- `src/pipeline/` — Pipeline orchestration and stages
- `src/training/` — Training strategies and orchestration
- `src/providers/` — GPU provider implementations (single_node, RunPod)
- `src/config/` — Configuration schemas (Pydantic v2)
- `src/evaluation/` — Model evaluation plugins
- `src/data/` — Dataset handling and validation
- `src/reports/` — Report generation plugins
- `src/tui/` — Terminal UI (Textual)

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- For bugs, include relevant log output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
