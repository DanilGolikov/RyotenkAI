"""Sentinel: ``detail=`` of typed errors must not interpolate PII variables.

Phase-G hardening (2026-05-16). Developers can accidentally write::

    raise ConfigInvalidError(detail=f"user {user.email} bad config")

The ``email`` value then flies into:
* the wire payload (RFC 9457 ``detail`` field, returned to the client),
* the structured log line via :meth:`RyotenkAIError.__str__` rendering,
* any downstream dashboard that groups on ``detail``.

The structured ``context={"email": user.email}`` parameter exists exactly
to carry PII-ish values out-of-band where they can be redacted (or
omitted entirely from the wire) without losing the diagnostic info in
logs. This sentinel pushes developers toward that pattern by AST-scanning
every concrete-error ``raise``/``construct`` site for f-strings whose
named substitutions look like PII.

Detection
---------
Flagged: any ``Call`` whose function name ends in ``Error`` (case-
sensitive, matches the :class:`RyotenkAIError` naming convention) whose
``detail=`` keyword is an f-string referencing a name from
:data:`_PII_VARIABLE_NAMES` (either as a bare ``Name``, an attribute
chain ending in one of those names, or a subscript ending in a
PII-named string literal key).

Not flagged: literal strings, calls that don't pass ``detail=``,
classes whose name doesn't end in ``Error``, or f-strings that
interpolate non-PII names.

False positives go in
:file:`tests/_lint/pii_in_errors_allowlist.yaml` until the call site
is migrated to the structured ``context=`` form.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
_ALLOWLIST_FILE = Path(__file__).parent / "pii_in_errors_allowlist.yaml"


# Variable / attribute / subscript-key names that, when interpolated
# into ``detail=`` of an Error constructor, indicate PII leakage.
# UPPER vs lower-case is normalised by ``.lower()`` before checking.
_PII_VARIABLE_NAMES: frozenset[str] = frozenset({
    "email",
    "user_email",
    "useremail",
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "api_token",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "secret",
    "client_secret",
    "credit_card",
    "creditcard",
    "card_number",
    "cardnumber",
    "cvv",
    "ssn",
    "social_security",
    "phone",
    "phone_number",
    "phonenumber",
})


def _load_allowlist() -> list[dict]:
    raw = yaml.safe_load(_ALLOWLIST_FILE.read_text(encoding="utf-8"))
    entries = raw.get("entries") or []
    return entries


def _is_allowlisted(rel_path: str, lineno: int, allowlist: list[dict]) -> bool:
    for entry in allowlist:
        if entry.get("path") != rel_path:
            continue
        if "lineno" in entry and entry["lineno"] != lineno:
            continue
        return True
    return False


def _looks_like_pii_name(name: str) -> bool:
    """Return True if ``name`` lower-cased equals or ends with a PII token.

    Conservative: ``user_email_addr`` matches because it ends with
    ``email_addr``? No — we match the exact lowered name first, then
    suffix-match against the canonical set with ``_`` boundary. This
    catches ``user_email`` (exact), ``customer_password`` (ends with
    ``_password``) without catching unrelated ``token_count``
    (``token`` is a prefix not a suffix).
    """
    lowered = name.lower()
    if lowered in _PII_VARIABLE_NAMES:
        return True
    # Suffix match with underscore boundary, e.g. ``hf_token`` ends with
    # ``_token`` (which lowercased equals one of the PII names).
    for canonical in _PII_VARIABLE_NAMES:
        if lowered.endswith("_" + canonical):
            return True
    return False


def _extract_referenced_names(node: ast.AST) -> Iterable[str]:
    """Walk an f-string node and yield every name-shaped reference.

    Yields:
    * ``ast.Name.id`` for bare names like ``email``,
    * the last segment of an ``ast.Attribute`` chain like
      ``user.email`` → ``"email"``,
    * the string-constant key of a subscript like ``d["password"]`` →
      ``"password"``.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            yield sub.id
        elif isinstance(sub, ast.Attribute):
            yield sub.attr
        elif isinstance(sub, ast.Subscript):
            # Only flag literal-string keys: ``d["email"]``. Computed
            # keys (``d[k]``) are out of scope — the static analyser
            # cannot know what ``k`` is.
            slc = sub.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                yield slc.value


def _is_error_constructor(node: ast.Call) -> bool:
    """Heuristic: does this call construct a ``*Error``?"""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id.endswith("Error")
    if isinstance(func, ast.Attribute):
        return func.attr.endswith("Error")
    return False


def _detail_kwarg(node: ast.Call) -> ast.AST | None:
    """Return the AST value of ``detail=...`` keyword, or ``None``."""
    for kw in node.keywords:
        if kw.arg == "detail":
            return kw.value
    return None


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, message)`` violations for ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return []

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_error_constructor(node):
            continue
        detail = _detail_kwarg(node)
        if detail is None:
            continue
        # Only f-strings can interpolate variables -- bare ``Constant``
        # strings are statically safe; ``BinOp``/``%`` formatting is
        # out of scope here (rare in this codebase). f-strings are
        # ``ast.JoinedStr``.
        if not isinstance(detail, ast.JoinedStr):
            continue
        for name in _extract_referenced_names(detail):
            if _looks_like_pii_name(name):
                violations.append((
                    detail.lineno,
                    f"f-string in detail= references {name!r}",
                ))
                break  # one finding per call is enough
    return violations


def _scan_packages() -> list[str]:
    """Walk ``packages/*/src/`` and collect non-allowlisted violations."""
    if not PACKAGES_ROOT.exists():
        return []
    allowlist = _load_allowlist()
    violations: list[str] = []
    for src_root in PACKAGES_ROOT.glob("*/src"):
        for py in src_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            rel = py.relative_to(PACKAGES_ROOT).as_posix()
            for lineno, msg in _scan_file(py):
                if _is_allowlisted(rel, lineno, allowlist):
                    continue
                violations.append(f"{rel}:{lineno}: {msg}")
    return violations


def test_no_pii_in_error_detail_fstrings() -> None:
    """No ``Error(detail=f"...{user.email}...")`` patterns in production.

    Migrate to the structured form::

        raise ConfigInvalidError(
            detail="user has invalid config",
            context={"email": user.email},
        )

    The structured ``context`` dict is separately sanitisable at the
    handler boundary and never lands in the human-readable ``detail``
    field of the wire payload.
    """
    violations = _scan_packages()
    assert not violations, (
        "f-string in Error(detail=...) interpolates PII-looking variable. "
        "Move the value to ``context={...}`` so the handler can sanitise "
        "it independently.\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# Self-tests — exercise the AST walker against synthetic inputs.
# ---------------------------------------------------------------------------


def test_sentinel_flags_email_in_fstring(tmp_path: Path) -> None:
    """``Error(detail=f"... {user.email} ...")`` is flagged."""
    bad = tmp_path / "bad_email.py"
    bad.write_text(
        "class Cfg: pass\n"
        "def boom(user):\n"
        "    raise ConfigInvalidError(detail=f'bad: {user.email}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "expected an email finding"
    assert "email" in findings[0][1]


def test_sentinel_flags_token_bare_name(tmp_path: Path) -> None:
    """``Error(detail=f"... {token} ...")`` (bare name) is flagged."""
    bad = tmp_path / "bad_token.py"
    bad.write_text(
        "def boom(token):\n"
        "    raise HFAuthFailedError(detail=f'tok: {token}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "expected a token finding"
    assert "token" in findings[0][1]


def test_sentinel_flags_subscript_key(tmp_path: Path) -> None:
    """``Error(detail=f"... {d['password']} ...")`` is flagged."""
    bad = tmp_path / "bad_subscript.py"
    bad.write_text(
        "def boom(d):\n"
        "    raise ProviderAuthFailedError(detail=f'p: {d[\"password\"]}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "expected a password finding"
    assert "password" in findings[0][1]


def test_sentinel_flags_suffix_named_token(tmp_path: Path) -> None:
    """``hf_token`` (suffix matches ``_token``) is flagged."""
    bad = tmp_path / "bad_suffix.py"
    bad.write_text(
        "def boom(hf_token):\n"
        "    raise HFAuthFailedError(detail=f't: {hf_token}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "expected a token suffix finding"


def test_sentinel_ignores_non_error_call(tmp_path: Path) -> None:
    """``log.info(detail=f"{user.email}")`` is NOT flagged — not an Error."""
    ok = tmp_path / "ok_logger.py"
    ok.write_text(
        "def boom(user, log):\n"
        "    log.info(detail=f'looking up {user.email}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings


def test_sentinel_ignores_literal_detail(tmp_path: Path) -> None:
    """``Error(detail="static string")`` is fine."""
    ok = tmp_path / "ok_literal.py"
    ok.write_text(
        "def boom():\n"
        "    raise ConfigInvalidError(detail='static text only')\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings


def test_sentinel_ignores_non_pii_fstring(tmp_path: Path) -> None:
    """``Error(detail=f"... {job_id} ...")`` is fine — ``job_id`` is not PII."""
    ok = tmp_path / "ok_nonpii.py"
    ok.write_text(
        "def boom(job_id):\n"
        "    raise JobStateInvalidError(detail=f'job: {job_id}')\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings


def test_sentinel_ignores_pii_in_context_dict(tmp_path: Path) -> None:
    """``Error(context={"email": x})`` is fine — context is the safe channel."""
    ok = tmp_path / "ok_context.py"
    ok.write_text(
        "def boom(user):\n"
        "    raise ConfigInvalidError(\n"
        "        detail='bad config',\n"
        "        context={'email': user.email},\n"
        "    )\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, (
        "context={'email': ...} is the canonical SAFE channel; must not flag"
    )


def test_allowlist_yaml_parses() -> None:
    """The allowlist YAML is well-formed and the loader returns a list."""
    entries = _load_allowlist()
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        assert "path" in entry
        assert "reason" in entry
