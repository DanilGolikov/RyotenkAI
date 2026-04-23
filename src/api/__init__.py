"""Web backend for RyotenkAI pipeline (FastAPI).

Architecture: Kubernetes-way Shared State. Backend is a sibling client to the
same file-based state store used by the CLI. It never wraps the CLI through
subprocess for read paths — it imports src.pipeline directly. Pipeline launches
are detached subprocesses (start_new_session=True) so the backend can restart
without orphaning runs.
"""
