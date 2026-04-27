"""Phase 14.B — single-node runner-side runtime package.

Mirrors :mod:`src.providers.runpod.runtime` for the single-node case
(local SSH host instead of RunPod cloud). Single-node has no cloud
lifecycle to act on, so the only thing in this package is a no-op
:class:`IPodLifecycleClient` impl that keeps the runner's terminal
hook happy without making any transport calls.
"""
