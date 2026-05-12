"""Convert bare ``MagicMock()`` data carriers to ``SimpleNamespace``.

This codemod is the workhorse of Phase 2A of the mock-elimination effort
(see ``docs/plans/mock-elimination-architecture.md``). It rewrites the
single highest-volume mock pattern in the test corpus:

* ``m = MagicMock()`` followed by ``m.foo = X; m.bar = Y`` (data carrier)
  becomes ``m = SimpleNamespace(foo=X, bar=Y)`` and the trailing
  attribute assignments are deleted.

It SKIPS cases where ``MagicMock`` semantics are actually needed:

* ``MagicMock()`` constructed with positional args, ``spec=``,
  ``spec_set=``, ``wraps=``, ``return_value=``, ``side_effect=``, or
  ``name=``.
* ``MagicMock()`` whose binding is later called (``m()``), used as a
  callable spy (``m.assert_called_with(...)``), or has
  ``.return_value`` / ``.side_effect`` mutated.
* Magic-method usage (``len(m)``, ``m[0]``, ``bool(m)``, ``iter(m)``)
  except for plain attribute access/assignment.
* ``MagicMock()`` passed as an argument to ``spec=`` / ``spec_set=`` /
  ``wraps=``.
* Inline ``MagicMock()`` (not bound to a name) is left untouched too —
  this codemod focuses on the assignment data-carrier pattern, which is
  the bulk of Phase 2A.  A second pass can handle inline cases later.

Run with ``--dry-run`` (default) to print a unified diff, or ``--apply``
to rewrite files in place.  Always run the test suite afterwards; the
codemod is conservative but not infallible.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst.codemod import Codemod, CodemodContext


# ---------------------------------------------------------------------------
# Heuristics — what disqualifies a variable from conversion
# ---------------------------------------------------------------------------

#: Kwargs to ``MagicMock(...)`` that flag the call as a *real* mock rather
#: than a data carrier.  Anything in this set means "user wants Mock
#: semantics", so we leave the assignment alone.
_MOCK_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "spec",
        "spec_set",
        "wraps",
        "return_value",
        "side_effect",
        "name",
        "_spec_class",
        "_spec_set",
        "_mock_name",
        "configure_mock",
    },
)

#: Mock-only attributes — touching any of these means the var is a spy/stub.
_MOCK_ONLY_ATTRS: frozenset[str] = frozenset(
    {
        "return_value",
        "side_effect",
        "configure_mock",
        "reset_mock",
        "mock_calls",
        "call_args",
        "call_args_list",
        "call_count",
        "called",
        "method_calls",
    },
)

#: Attribute prefixes that flag interaction tests.
_MOCK_ASSERT_PREFIXES: tuple[str, ...] = ("assert_",)


def _attribute_is_mock_only(attr_name: str) -> bool:
    """Return True if accessing ``var.<attr_name>`` indicates Mock semantics."""

    if attr_name in _MOCK_ONLY_ATTRS:
        return True
    return any(attr_name.startswith(prefix) for prefix in _MOCK_ASSERT_PREFIXES)


# ---------------------------------------------------------------------------
# Per-variable bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _CandidateRecord:
    """Tracks analysis state for a single ``var = MagicMock()`` site."""

    assign_node: cst.Assign
    target_name: str
    #: Attribute assignments ``var.foo = X`` collected for absorption,
    #: in source order.  Stops growing once a disqualifying use occurs.
    absorbed_attr_assigns: list[cst.Assign] = field(default_factory=list)
    #: Set when something disqualifies the variable from conversion.
    disqualified: bool = False
    #: True once the variable has been read or used in a non-assignment
    #: context.  After this, further ``var.foo = X`` assignments are no
    #: longer eligible for absorption (they happen *after* code has
    #: already started consuming the namespace).
    consumed: bool = False
    #: True once the absorption window has closed.  Absorption only
    #: continues while the statements immediately following
    #: ``var = MagicMock()`` are themselves ``var.attr = X``
    #: assignments.  Any other statement (function def, expression,
    #: etc.) closes the window — assignments after that point cannot
    #: be folded because they may reference names not yet defined at
    #: the original line.
    absorption_open: bool = True


# ---------------------------------------------------------------------------
# Pass 1 — analyse: build the candidate map
# ---------------------------------------------------------------------------


class _AnalysisVisitor(cst.CSTVisitor):
    """Identify ``var = MagicMock()`` candidates and their usage pattern.

    Two-pass within a single traversal:

    1. Whenever we see an :class:`cst.Assign` with a single ``Name`` target
       whose value is a *clean* ``MagicMock()`` call, register a candidate.
    2. For every subsequent reference to a registered name, classify:

       * ``Attribute(value=Name(var), attr=Name(foo))`` on its own as an
         assignment target — record for absorption.
       * Anything else — disqualify or mark consumed.
    """

    def __init__(self) -> None:
        super().__init__()
        # name -> latest active candidate.  Reassignment overwrites.
        self._candidates: dict[str, _CandidateRecord] = {}
        # All candidates discovered (active or retired), keyed by id().
        self._all: dict[int, _CandidateRecord] = {}
        # Stack of nodes we're currently inside; used to ignore the
        # candidate's own assignment node when scanning usage.
        self._suppress: set[int] = set()
        # Pre-computed sets of Name node ids.
        # - ``_attr_position``: Name appears as ``Attribute.attr`` or a
        #   kwarg ``keyword=``.  Not a variable reference.
        # - ``_safe_uses``: Name appears at the head of an attribute
        #   chain (``name.foo``) or as an assignment target.
        # Everything else is a "risky use" — disqualifies the candidate
        # if seen.
        self._attr_position: set[int] = set()
        self._safe_uses: set[int] = set()
        #: ``id(SimpleStatementLine)`` for assignments that are
        #: physically *contiguous* with the original ``var = MagicMock()``
        #: line and the candidate's accumulating ``var.attr = X`` block.
        #: Only assignments in this set are absorbable.
        self._contiguous_attr_lines: set[int] = set()
        #: Current SimpleStatementLine id being visited (or None).
        self._current_stmt_line_id: int | None = None

    def precompute(self, tree: cst.CSTNode) -> None:
        """Populate the alias-risk classification before the main walk."""

        tree.visit(
            _AliasUseScanner(self._attr_position, self._safe_uses),
        )
        tree.visit(
            _ContiguousAttrLineScanner(self._contiguous_attr_lines),
        )

    # --- accessors -------------------------------------------------------

    @property
    def candidates(self) -> Iterable[_CandidateRecord]:
        return self._all.values()

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _is_bare_magicmock(call: cst.Call) -> bool:
        """Return True for ``MagicMock()`` with no args / only safe kwargs.

        A *clean* MagicMock has zero positional args and no ``spec=`` /
        ``return_value=`` style kwargs.  Plain attribute-style kwargs
        like ``MagicMock(foo=1)`` are *also* treated as clean — they map
        straight onto ``SimpleNamespace(foo=1)``.
        """

        func = call.func
        if isinstance(func, cst.Name):
            if func.value != "MagicMock":
                return False
        elif isinstance(func, cst.Attribute):
            if func.attr.value != "MagicMock":
                return False
        else:
            return False

        for arg in call.args:
            # Positional arg (no keyword=) means we don't know the
            # semantics; skip conversion.
            if arg.keyword is None and arg.star == "":
                return False
            if arg.star in ("*", "**"):
                return False
            kw = arg.keyword
            if isinstance(kw, cst.Name) and kw.value in _MOCK_ONLY_KWARGS:
                return False
        return True

    # --- assignment detection -------------------------------------------

    def visit_SimpleStatementLine(
        self, node: cst.SimpleStatementLine,
    ) -> bool | None:
        # Track current statement line so child Assigns know whether
        # they're in a contiguous block with the candidate's origin.
        self._current_stmt_line_id = id(node)
        return None

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        # Single-target ``name = MagicMock()`` is what we capture.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0].target, cst.Name)
            and isinstance(node.value, cst.Call)
            and self._is_bare_magicmock(node.value)
        ):
            name = node.targets[0].target.value
            # Suppress this Name reference from the usage scan below.
            self._suppress.add(id(node.targets[0].target))
            self._suppress.add(id(node.value))
            record = _CandidateRecord(assign_node=node, target_name=name)
            self._candidates[name] = record
            self._all[id(node)] = record
            return True

        # Attribute assignment targets — three sub-cases:
        #   (a) ``name.attr = X``                 → absorb candidate
        #   (b) ``name.foo.bar = X`` (any nested) → DISQUALIFY name
        #   (c) ``name.foo.return_value = X``    → DISQUALIFY name
        if len(node.targets) == 1 and isinstance(
            node.targets[0].target, cst.Attribute,
        ):
            target_attr: cst.Attribute = node.targets[0].target
            root_name = _attribute_root_name(target_attr)
            if root_name is not None:
                record = self._candidates.get(root_name)
                if record is not None and not record.disqualified:
                    # Sub-case (a): exactly ``name.attr = X``.
                    if isinstance(target_attr.value, cst.Name):
                        attr = target_attr.attr.value
                        if _attribute_is_mock_only(attr):
                            record.disqualified = True
                            return True
                        if _contains_name(node.value, root_name):
                            record.consumed = True
                            return True
                        if record.consumed:
                            # Post-use mutation; leave in place.  We
                            # still need to mark the var as touched so
                            # the import logic knows the conversion is
                            # safe.
                            return True
                        # The current statement must be physically
                        # contiguous with the previous absorbed
                        # statement (the ``var = MagicMock()`` itself
                        # or another ``var.attr = X`` we already took).
                        # If anything else came between, absorption is
                        # closed — leave the assignment as-is.
                        if (
                            self._current_stmt_line_id is None
                            or self._current_stmt_line_id
                            not in self._contiguous_attr_lines
                        ):
                            record.absorption_open = False
                            return True
                        if not record.absorption_open:
                            return True
                        record.absorbed_attr_assigns.append(node)
                        # Suppress the Name inside the LHS attribute
                        # from counting as a "read" of var.
                        self._suppress.add(id(target_attr.value))
                        return True
                    # Sub-cases (b) and (c): nested assignment target.
                    # SimpleNamespace can't auto-create the intermediate
                    # attribute, so we cannot safely convert.
                    record.disqualified = True
                    return True
        return None


    # --- usage tracking --------------------------------------------------

    def visit_Name(self, node: cst.Name) -> None:
        if id(node) in self._suppress:
            return
        if id(node) in self._attr_position:
            # ``foo.NAME`` or ``f(NAME=value)`` — NAME here is a label,
            # not a variable reference.  Ignore.
            return
        record = self._candidates.get(node.value)
        if record is None or record.disqualified:
            return
        # Mark consumed regardless of safe/risky.  Any use of the
        # variable beyond an absorbable attribute assignment stops
        # the kwarg gathering.
        record.consumed = True

    def visit_Call(self, node: cst.Call) -> bool | None:
        # ``var(...)`` — direct call disqualifies.
        if isinstance(node.func, cst.Name):
            record = self._candidates.get(node.func.value)
            if record is not None and not record.disqualified:
                record.disqualified = True
        # ``var.method(...)``: the Name ref inside func is fine for
        # SimpleNamespace, but if the resulting attribute call later
        # gets ``.return_value = ...`` etc, that's spotted elsewhere.
        # We *don't* disqualify on var.foo() because SimpleNamespace
        # works as long as foo itself is callable (e.g. AsyncMock).
        # But we still mark "consumed" so attribute assigns afterwards
        # are not absorbed.
        if isinstance(node.func, cst.Attribute) and isinstance(
            node.func.value, cst.Name,
        ):
            record = self._candidates.get(node.func.value.value)
            if record is not None and not record.disqualified:
                record.consumed = True
        # ``monkeypatch.setattr(target, attr_name, OUR_CANDIDATE)``,
        # ``setattr(target, name, OUR_CANDIDATE)``, or
        # ``patch.object(target, "attr", new=OUR_CANDIDATE)``: the
        # candidate becomes the production binding for ``target.attr``.
        # Downstream code in the system under test may use Mock-only
        # semantics on ``target.attr``, so keep Mock semantics.
        if _is_setattr_like_call(node.func):
            for arg in node.args:
                if isinstance(arg.value, cst.Name):
                    record = self._candidates.get(arg.value.value)
                    if record is not None and not record.disqualified:
                        record.disqualified = True
        return None

    def visit_Attribute(self, node: cst.Attribute) -> bool | None:
        # ``var.return_value`` / ``var.assert_called*`` — disqualify.
        # Also catches the chained leaf ``var.foo.return_value`` because
        # the *whole* attribute chain is visited top-down and the root
        # of the chain is what we key on.
        root_name = _attribute_root_name(node)
        if root_name is None:
            return None
        record = self._candidates.get(root_name)
        if record is None or record.disqualified:
            return None
        if _attribute_is_mock_only(node.attr.value):
            record.disqualified = True
        return None

    def visit_Arg(self, node: cst.Arg) -> bool | None:
        # ``MagicMock(spec=var)`` — using var as a spec source is fine,
        # but it definitely *consumes* the variable.  We don't need to
        # do anything special here; the inner Name visit will mark
        # ``consumed = True``.  Likewise for passing var as a normal
        # argument.
        return None


# ---------------------------------------------------------------------------
# Helper — check if a CST subtree references a Name by string
# ---------------------------------------------------------------------------


class _ContiguousAttrLineScanner(cst.CSTVisitor):
    """Mark ``SimpleStatementLine`` ids that are physically contiguous
    with a ``var = MagicMock()`` and the candidate's accumulating
    ``var.attr = X`` block.

    Absorption can only proceed for assignments in these blocks; any
    interleaved statement (function def, expression, other assignment)
    closes the absorption window, because the absorbed RHS would be
    evaluated *at the original line* — which may reference names not
    yet defined.
    """

    def __init__(self, sink: set[int]) -> None:
        super().__init__()
        self._sink = sink

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> bool | None:
        self._scan_body(list(node.body))
        return None

    def visit_Module(self, node: cst.Module) -> bool | None:
        self._scan_body(list(node.body))
        return None

    def _scan_body(self, body: list[cst.BaseStatement]) -> None:
        # A "run" starts when we see ``var = MagicMock()`` and continues
        # as long as the next statement is ``var.attr = X`` (any attr).
        # Multiple candidates can be interleaved in their own runs —
        # we track them per-name.
        active: dict[str, cst.SimpleStatementLine] = {}
        for stmt in body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                # E.g. function def, for-loop, with-block.  Close all
                # active runs.
                active.clear()
                continue
            consumed_active = False
            for small in stmt.body:
                if isinstance(small, cst.Assign):
                    if (
                        len(small.targets) == 1
                        and isinstance(small.targets[0].target, cst.Name)
                        and isinstance(small.value, cst.Call)
                        and _is_bare_magicmock_call(small.value)
                    ):
                        name = small.targets[0].target.value
                        active[name] = stmt
                        consumed_active = True
                        continue
                    if (
                        len(small.targets) == 1
                        and isinstance(small.targets[0].target, cst.Attribute)
                    ):
                        target_attr = small.targets[0].target
                        root = _attribute_root_name(target_attr)
                        if root is not None and root in active:
                            # Contiguous with this candidate's run.
                            self._sink.add(id(stmt))
                            consumed_active = True
                            continue
                # Any other small statement (expr, multi-assign, etc.)
                # closes all active runs.
                active.clear()
                break
            if not consumed_active:
                active.clear()


def _is_bare_magicmock_call(call: cst.Call) -> bool:
    """Mirror of ``_AnalysisVisitor._is_bare_magicmock`` for module scope."""

    func = call.func
    if isinstance(func, cst.Name):
        if func.value != "MagicMock":
            return False
    elif isinstance(func, cst.Attribute):
        if func.attr.value != "MagicMock":
            return False
    else:
        return False
    for arg in call.args:
        if arg.keyword is None and arg.star == "":
            return False
        if arg.star in ("*", "**"):
            return False
        kw = arg.keyword
        if isinstance(kw, cst.Name) and kw.value in _MOCK_ONLY_KWARGS:
            return False
    return True


class _AliasUseScanner(cst.CSTVisitor):
    """Classify every Name node by *role* in its parent context.

    The result is three sets keyed by ``id(name_node)``:

    * ``attr_position`` — Names used as ``Attribute.attr`` or as the
      ``keyword=`` part of a kwarg or import alias.  These are not
      variable references at all and should be ignored.
    * ``safe_uses`` — Names used in roles that do NOT alias the value
      to another location: the head of an attribute chain
      (``name.foo``), the target of an assignment (``name = ...``),
      or a delete/global/nonlocal target.
    * (implicit) ``risky_uses`` — every other Name.  Examples: passed
      as an argument, used as the RHS of an assignment, returned,
      yielded, compared.  These risk aliasing the value and disqualify
      a candidate.
    """

    def __init__(
        self,
        attr_position: set[int],
        safe_uses: set[int],
    ) -> None:
        super().__init__()
        self._attr_position = attr_position
        self._safe_uses = safe_uses

    def visit_Attribute(self, node: cst.Attribute) -> bool | None:
        # ``node.attr`` is a Name acting as an attribute *name*, not a
        # variable reference.
        self._attr_position.add(id(node.attr))
        # The head of the attribute chain (deepest .value if it's a
        # Name) is a *safe use* of that variable — accessing a member,
        # no aliasing.
        cur: cst.BaseExpression = node
        while isinstance(cur, cst.Attribute):
            cur = cur.value
        if isinstance(cur, cst.Name):
            self._safe_uses.add(id(cur))
        return None

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        if isinstance(node.target, cst.Name):
            self._safe_uses.add(id(node.target))
        return None

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        if isinstance(node.target, cst.Name):
            self._safe_uses.add(id(node.target))
        return None

    def visit_AugAssign(self, node: cst.AugAssign) -> bool | None:
        # ``x += ...`` — reads x then writes back.  Not safe.
        return None

    def visit_Param(self, node: cst.Param) -> bool | None:
        # Function parameters: the parameter name is a binding, not a
        # use of an outer variable.  We treat its Name as ``safe_uses``
        # so it doesn't disqualify a same-named candidate.
        if isinstance(node.name, cst.Name):
            self._safe_uses.add(id(node.name))
        return None

    def visit_Arg(self, node: cst.Arg) -> bool | None:
        # ``keyword=value`` — the keyword Name is just a label.
        if node.keyword is not None and isinstance(node.keyword, cst.Name):
            self._attr_position.add(id(node.keyword))
        return None

    def visit_ImportAlias(self, node: cst.ImportAlias) -> bool | None:
        # Imported names are not "uses" of outer variables.
        if isinstance(node.name, cst.Name):
            self._attr_position.add(id(node.name))
        if node.asname is not None and isinstance(node.asname.name, cst.Name):
            self._attr_position.add(id(node.asname.name))
        return None


def _is_setattr_like_call(func: cst.BaseExpression) -> bool:
    """Return True if ``func`` is a callable that aliases a value to
    a target attribute (``setattr``, ``monkeypatch.setattr``,
    ``patch.object``).

    These calls take a value (usually our candidate) and bind it to
    ``target.attr`` so downstream code uses the candidate via that
    alias.  Mock-only operations on ``target.attr`` would then touch
    the candidate, which Codemod analysis can't see — so we disqualify
    the candidate to preserve ``MagicMock`` semantics.
    """

    if isinstance(func, cst.Name) and func.value == "setattr":
        return True
    if isinstance(func, cst.Attribute):
        name = func.attr.value
        if name in {"setattr", "object", "dict", "multiple"}:
            return True
    return False


def _attribute_root_name(attr: cst.Attribute) -> str | None:
    """Walk ``attr.value`` down until we hit a Name; return that name or None.

    For ``a.b.c.d`` this returns ``"a"``.  For ``foo().b`` it returns
    ``None`` because the root is a Call, not a Name.
    """

    cur: cst.BaseExpression = attr
    while isinstance(cur, cst.Attribute):
        cur = cur.value
    if isinstance(cur, cst.Name):
        return cur.value
    return None


def _contains_name(node: cst.CSTNode, name: str) -> bool:
    class _Finder(cst.CSTVisitor):
        def __init__(self) -> None:
            self.hit = False

        def visit_Name(self, n: cst.Name) -> None:
            if n.value == name:
                self.hit = True

    visitor = _Finder()
    node.visit(visitor)
    return visitor.hit


# ---------------------------------------------------------------------------
# Pass 2 — rewrite: replace MagicMock() with SimpleNamespace(...)
# ---------------------------------------------------------------------------


class _RewriteTransformer(cst.CSTTransformer):
    """Apply the conversions decided by :class:`_AnalysisVisitor`."""

    def __init__(self, records: dict[int, _CandidateRecord]) -> None:
        super().__init__()
        self._records = records
        # Map id(original_assign) -> list of absorbed attribute kwargs.
        # Also a set of ids of the absorbed attr-assignment nodes so we
        # can delete them on visit.
        self._absorbed_ids: set[int] = set()
        self._converted: set[int] = set()
        self.changed_count: int = 0
        for record in records.values():
            if record.disqualified:
                continue
            for assign in record.absorbed_attr_assigns:
                self._absorbed_ids.add(id(assign))

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign,
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        # Delete absorbed attribute assignments.
        if id(original_node) in self._absorbed_ids:
            return cst.RemoveFromParent()

        # Rewrite candidate ``var = MagicMock()`` -> SimpleNamespace.
        record = self._records.get(id(original_node))
        if record is None or record.disqualified:
            return updated_node

        # Gather kwargs: first, any kwargs that were already on
        # ``MagicMock(foo=1)``; then absorbed attribute assignments.
        new_args: list[cst.Arg] = []
        assert isinstance(updated_node.value, cst.Call)
        for arg in updated_node.value.args:
            new_args.append(arg)
        for i, attr_assign in enumerate(record.absorbed_attr_assigns):
            assert isinstance(attr_assign.targets[0].target, cst.Attribute)
            attr_name = attr_assign.targets[0].target.attr.value
            # Build ``attr_name=value``.
            new_args.append(
                cst.Arg(
                    keyword=cst.Name(attr_name),
                    value=attr_assign.value,
                    equal=cst.AssignEqual(
                        whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.SimpleWhitespace(""),
                    ),
                    comma=(
                        cst.Comma(
                            whitespace_after=cst.SimpleWhitespace(" "),
                        )
                        if i < len(record.absorbed_attr_assigns) - 1
                        else cst.MaybeSentinel.DEFAULT
                    ),
                ),
            )
        # Ensure all but the last arg have a comma+space.
        new_args = _normalize_arg_commas(new_args)

        new_call = cst.Call(
            func=cst.Name("SimpleNamespace"),
            args=new_args,
        )
        self._converted.add(id(original_node))
        self.changed_count += 1
        return updated_node.with_changes(value=new_call)


def _normalize_arg_commas(args: list[cst.Arg]) -> list[cst.Arg]:
    """Ensure every arg except the last has a trailing comma."""

    out: list[cst.Arg] = []
    last = len(args) - 1
    for i, arg in enumerate(args):
        if i < last:
            comma = (
                arg.comma
                if isinstance(arg.comma, cst.Comma)
                else cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
            )
            out.append(arg.with_changes(comma=comma))
        else:
            out.append(arg.with_changes(comma=cst.MaybeSentinel.DEFAULT))
    return out


# ---------------------------------------------------------------------------
# Pass 3 — top-level: SimpleNamespace import management
# ---------------------------------------------------------------------------


class _ImportFixer(cst.CSTTransformer):
    """Add ``from types import SimpleNamespace`` if needed, and drop
    ``MagicMock`` from ``unittest.mock`` imports when it's no longer used.
    """

    def __init__(self, *, needs_simplenamespace: bool, magicmock_still_used: bool) -> None:
        super().__init__()
        self._needs_sn = needs_simplenamespace
        self._mm_still_used = magicmock_still_used
        self._has_sn_import = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
        if (
            isinstance(node.module, cst.Attribute | cst.Name)
            and _module_dotted_name(node.module) == "types"
            and isinstance(node.names, Sequence)
        ):
            for alias in node.names:
                if (
                    isinstance(alias, cst.ImportAlias)
                    and isinstance(alias.name, cst.Name)
                    and alias.name.value == "SimpleNamespace"
                    and alias.asname is None
                ):
                    self._has_sn_import = True
        return None

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        # ``from unittest.mock import MagicMock, ...`` — drop MagicMock
        # if it's no longer used.
        if not (
            isinstance(updated_node.module, cst.Attribute | cst.Name)
            and _module_dotted_name(updated_node.module) == "unittest.mock"
        ):
            return updated_node
        if self._mm_still_used:
            return updated_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        kept: list[cst.ImportAlias] = []
        for alias in updated_node.names:
            assert isinstance(alias, cst.ImportAlias)
            if (
                isinstance(alias.name, cst.Name)
                and alias.name.value == "MagicMock"
                and alias.asname is None
            ):
                continue
            kept.append(alias)
        if not kept:
            return cst.RemoveFromParent()
        kept = list(_normalize_alias_commas(kept))
        return updated_node.with_changes(names=kept)

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module,
    ) -> cst.Module:
        if not self._needs_sn or self._has_sn_import:
            return updated_node
        # Insert ``from types import SimpleNamespace`` after the last
        # ``from __future__`` import, after the docstring, or at the top.
        body = list(updated_node.body)
        insert_idx, needs_blank_before = _find_import_insertion_index(body)
        leading_lines: tuple[cst.EmptyLine, ...] = ()
        if needs_blank_before:
            leading_lines = (cst.EmptyLine(),)
        new_import = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Name("types"),
                    names=[cst.ImportAlias(name=cst.Name("SimpleNamespace"))],
                ),
            ],
            leading_lines=leading_lines,
        )
        body.insert(insert_idx, new_import)
        return updated_node.with_changes(body=body)


def _module_dotted_name(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_module_dotted_name(node.value)}.{node.attr.value}"
    return ""


def _normalize_alias_commas(aliases: list[cst.ImportAlias]) -> Iterator[cst.ImportAlias]:
    last = len(aliases) - 1
    for i, alias in enumerate(aliases):
        if i < last:
            yield alias.with_changes(
                comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
            )
        else:
            yield alias.with_changes(comma=cst.MaybeSentinel.DEFAULT)


def _find_import_insertion_index(
    body: list[cst.BaseStatement],
) -> tuple[int, bool]:
    """Find a good place to insert a new ``from types import …`` import.

    Returns ``(index, needs_blank_line_before)``.  Preference order:
    just after the last ``from __future__`` import (with a PEP 8 blank
    separating the ``__future__`` group from the stdlib group); or
    just after the module docstring (with a PEP 8 blank line); or at
    the very top (no extra blank).
    """

    last_future_idx = -1
    docstring_idx = -1
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if (
                    isinstance(small, cst.ImportFrom)
                    and isinstance(small.module, cst.Name)
                    and small.module.value == "__future__"
                ):
                    last_future_idx = i
                elif (
                    docstring_idx == -1
                    and isinstance(small, cst.Expr)
                    and isinstance(small.value, cst.SimpleString)
                ):
                    docstring_idx = i
    if last_future_idx >= 0:
        return last_future_idx + 1, True
    if docstring_idx >= 0:
        return docstring_idx + 1, True
    return 0, False


# ---------------------------------------------------------------------------
# Top-level Codemod
# ---------------------------------------------------------------------------


class MagicMockToSimpleNamespaceCodemod(Codemod):
    """Replace bare ``MagicMock()`` data carriers with ``SimpleNamespace``.

    Detection heuristic: a ``MagicMock()`` call is convertible if and only
    if every use of the assigned name is one of:

    * an attribute assignment (``var.foo = X``) on a non-mock-only name
    * an attribute read or attribute call (``var.foo``, ``var.foo()``)
    * passing ``var`` as a plain argument (read)

    and *no* use is:

    * calling ``var`` directly,
    * reading/writing ``var.return_value``, ``var.side_effect``, or
      ``var.assert_called*``,
    * positional/spec/``return_value=``-style construction arguments.

    Eligible attribute assignments (``var.foo = X``) that appear before
    any *consuming* use are absorbed into the ``SimpleNamespace`` kwargs.
    Mutations of ``var.foo = X`` *after* the variable is first used are
    left in place (SimpleNamespace supports attribute writes natively).
    """

    DESCRIPTION = "Replace bare MagicMock() data carriers with SimpleNamespace"

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)
        self.changed_count: int = 0

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        # Pass 1 — analyse.
        analyser = _AnalysisVisitor()
        analyser.precompute(tree)
        tree.visit(analyser)

        # If nothing to do, bail early.
        records_by_id = {
            id(rec.assign_node): rec for rec in analyser.candidates
        }
        if not any(not r.disqualified for r in records_by_id.values()):
            return tree

        # Pass 2 — rewrite.
        rewriter = _RewriteTransformer(records_by_id)
        tree = tree.visit(rewriter)
        self.changed_count = rewriter.changed_count

        # Pass 3 — fix imports.
        # SimpleNamespace is needed iff we actually rewrote something.
        # MagicMock is still used iff any MagicMock(...) call remains
        # anywhere (including ones we skipped).
        remains = _MagicMockUsageScanner()
        tree.visit(remains)
        fixer = _ImportFixer(
            needs_simplenamespace=self.changed_count > 0,
            magicmock_still_used=remains.found,
        )
        tree = tree.visit(fixer)
        return tree


class _MagicMockUsageScanner(cst.CSTVisitor):
    """Detect whether any ``MagicMock`` reference remains outside imports."""

    def __init__(self) -> None:
        super().__init__()
        self.found = False
        self._in_import = 0

    def visit_Import(self, node: cst.Import) -> bool | None:
        self._in_import += 1
        return None

    def leave_Import(self, node: cst.Import) -> None:
        self._in_import -= 1

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
        self._in_import += 1
        return None

    def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
        self._in_import -= 1

    def visit_Name(self, node: cst.Name) -> None:
        if self._in_import:
            return
        if node.value == "MagicMock":
            self.found = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_target_files(paths: Sequence[str]) -> Iterator[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for sub in sorted(p.rglob("*.py")):
                yield sub


def _run_on_file(path: Path, *, apply: bool) -> tuple[int, str]:
    """Run the codemod on a single file. Return ``(changed_count, diff)``."""

    src = path.read_text()
    context = CodemodContext(filename=str(path))
    codemod = MagicMockToSimpleNamespaceCodemod(context)
    tree = cst.parse_module(src)
    new_tree = codemod.transform_module_impl(tree)
    new_src = new_tree.code
    if new_src == src:
        return 0, ""
    diff = "".join(
        difflib.unified_diff(
            src.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
        ),
    )
    if apply:
        path.write_text(new_src)
    return codemod.changed_count, diff


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.codemods.magicmock_to_simplenamespace",
        description=MagicMockToSimpleNamespaceCodemod.DESCRIPTION,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to transform (recursive).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write transformed source back to disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print unified diffs but do not write (default if --apply absent).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print diffs, only the summary.",
    )
    args = parser.parse_args(argv)

    apply = args.apply and not args.dry_run

    total_files = 0
    changed_files = 0
    total_changes = 0
    for path in _iter_target_files(args.paths):
        total_files += 1
        try:
            n, diff = _run_on_file(path, apply=apply)
        except cst.ParserSyntaxError as e:
            print(f"SKIP {path}: parse error: {e}", file=sys.stderr)
            continue
        if n == 0:
            continue
        changed_files += 1
        total_changes += n
        if not args.quiet:
            sys.stdout.write(diff)
    sys.stdout.write(
        f"\n[magicmock_to_simplenamespace] "
        f"scanned {total_files} files, "
        f"{'rewrote' if apply else 'would rewrite'} {changed_files} files "
        f"({total_changes} MagicMock data carriers converted).\n",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
