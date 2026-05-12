"""Sidecar HTTP wrappers around in-process fakes.

Each sidecar is a Python module with a ``__main__`` that takes
``--port`` / ``--clock`` / ``--seed``, builds a FastAPI app via
:func:`tests._harness.stack.sidecars._base.make_app`, and runs uvicorn.
"""
