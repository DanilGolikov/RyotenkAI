"""Lint sentinels for the greenfield tree.

Each sentinel is a pytest test that walks the AST of ``tests/`` and fails
on a forbidden pattern (mocking Protocols, …).
"""
