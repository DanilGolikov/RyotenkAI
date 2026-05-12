"""LibCST codemods for the mock-elimination effort.

Each codemod is a :class:`libcst.codemod.Codemod` subclass with:

- A ``TRANSFORM`` class attribute (the codemod class itself, picked up
  by :class:`libcst.codemod.CodemodTest`).
- Test cases under ``scripts/codemods/test_cases/<scenario>/{before,after}.py``.
- A CLI entry: ``python -m scripts.codemods.<name> <files...> [--apply]``.

See ``docs/plans/mock-elimination-architecture.md`` — Phase 2A.
"""
