"""Convert ``patch.dict("os.environ", {...})`` to ``monkeypatch.setenv``.

This codemod is the Phase 2B mechanical conversion of the mock-elimination
effort (see ``docs/plans/mock-elimination-architecture.md``).  It rewrites
``unittest.mock.patch.dict`` calls that target ``os.environ`` into
pytest's native ``monkeypatch`` fixture:

* ``with patch.dict("os.environ", {"FOO": "bar"}): body``
  becomes ``monkeypatch.setenv("FOO", "bar")`` followed by ``body``
  (the function gains a ``monkeypatch`` parameter if it didn't already
  have one).
* ``@patch.dict("os.environ", {"FOO": "bar"}) def test_x(): ...``
  becomes ``def test_x(monkeypatch): monkeypatch.setenv("FOO", "bar"); ...``.
* Compound with-statements (``with patch.dict(...), other():``) have
  just the matching item peeled off; the rest of the with is preserved.

SKIPPED forms (left untouched with a ``# TODO(codemod)`` comment):

* ``patch.dict(..., clear=True)`` — needs context-specific cleanup logic.
* ``with patch.dict(...) as env_dict:`` — the bound name is referenced
  inside the body; ``monkeypatch.setenv`` is statement-based.
* Second argument that is not a plain ``Dict(...)`` literal (e.g. a
  variable like ``patch.dict("os.environ", env_without_hf)``) — the
  set of keys is unknown at codemod time.
* Non-string targets (anything other than the literal string
  ``"os.environ"`` or the attribute ``os.environ``).

The codemod also drops the ``from unittest.mock import patch`` import
when ``patch`` is no longer referenced anywhere in the module.

Run with ``--dry-run`` (default) to print a unified diff, or ``--apply``
to rewrite files in place.  Always run the test suite afterwards via
``scripts/codemods/apply_with_revert.py``.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import libcst as cst
from libcst.codemod import Codemod, CodemodContext


# ---------------------------------------------------------------------------
# Helpers — recognising the patch.dict("os.environ", ...) shape
# ---------------------------------------------------------------------------


_TODO_COMMENT = "# TODO(codemod): manual review needed for clear=True / with-as binding"


def _is_patch_dict_call(call: cst.Call) -> bool:
    """Return True if ``call`` syntactically is ``patch.dict(...)``.

    Matches the common case ``patch.dict(...)`` (where ``patch`` is the
    name in scope, imported from ``unittest.mock``).
    """

    func = call.func
    if not isinstance(func, cst.Attribute):
        return False
    if func.attr.value != "dict":
        return False
    if not isinstance(func.value, cst.Name):
        return False
    return func.value.value == "patch"


def _is_os_environ_target(arg_value: cst.BaseExpression) -> bool:
    """Return True if the first arg of ``patch.dict`` is ``os.environ``.

    Accepts either the string-literal form ``"os.environ"`` (mock's
    string-target API) or the attribute form ``os.environ``.
    """

    if isinstance(arg_value, cst.SimpleString):
        # SimpleString.value preserves the surrounding quotes; strip.
        # Treat both ``"os.environ"`` and ``'os.environ'`` as equivalent.
        return arg_value.evaluated_value == "os.environ"
    if isinstance(arg_value, cst.Attribute):
        if (
            isinstance(arg_value.value, cst.Name)
            and arg_value.value.value == "os"
            and arg_value.attr.value == "environ"
        ):
            return True
    return False


def _has_clear_true(call: cst.Call) -> bool:
    """Return True if the call has ``clear=True`` keyword argument."""

    for arg in call.args:
        if (
            arg.keyword is not None
            and isinstance(arg.keyword, cst.Name)
            and arg.keyword.value == "clear"
        ):
            v = arg.value
            if isinstance(v, cst.Name) and v.value == "True":
                return True
    return False


def _dict_literal_pairs(
    call: cst.Call,
) -> list[tuple[cst.BaseExpression, cst.BaseExpression]] | None:
    """Extract ``(key, value)`` pairs from the second positional argument.

    Returns ``None`` if the second arg is missing, isn't a ``Dict``
    literal, or contains non-key/value elements (``**spread``, etc.).
    """

    # First positional arg is target; second is the dict.
    positionals = [a for a in call.args if a.keyword is None and a.star == ""]
    if len(positionals) < 2:
        return None
    second = positionals[1].value
    if not isinstance(second, cst.Dict):
        return None
    pairs: list[tuple[cst.BaseExpression, cst.BaseExpression]] = []
    for element in second.elements:
        if not isinstance(element, cst.DictElement):
            # ``**spread`` etc. — bail out.
            return None
        pairs.append((element.key, element.value))
    return pairs


def _is_convertible_patch_dict(call: cst.Call) -> bool:
    """Return True if the codemod can safely convert this call."""

    if not _is_patch_dict_call(call):
        return False
    if not call.args:
        return False
    first = call.args[0]
    if first.keyword is not None or first.star != "":
        return False
    if not _is_os_environ_target(first.value):
        return False
    if _has_clear_true(call):
        return False
    pairs = _dict_literal_pairs(call)
    if pairs is None:
        return False
    # Every key must be a string literal — ``setenv`` takes a string name.
    for key, _ in pairs:
        if not isinstance(key, cst.SimpleString):
            return False
    return True


def _is_skip_candidate_patch_dict(call: cst.Call) -> bool:
    """Return True if the call targets os.environ but is unconvertible.

    These are the calls that should receive a ``# TODO(codemod)`` comment
    rather than be silently passed over (so a follow-up cleanup is
    visible in source).
    """

    if not _is_patch_dict_call(call):
        return False
    if not call.args:
        return False
    first = call.args[0]
    if first.keyword is not None or first.star != "":
        return False
    if not _is_os_environ_target(first.value):
        return False
    # If it would have been convertible, that's not a skip candidate.
    return not _is_convertible_patch_dict(call)


def _strip_trailing_comma(items: list[cst.WithItem]) -> list[cst.WithItem]:
    """Drop the trailing comma from the last WithItem.

    After removing items from a compound ``with (A, B, C):`` we must
    fix the comma of the new last item, which may still reference
    layout from a dropped item (causing trailing-whitespace lint
    violations).
    """

    if not items:
        return items
    out = list(items)
    last = out[-1]
    if isinstance(last.comma, cst.Comma):
        out[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return out


def _build_setenv_calls(
    pairs: list[tuple[cst.BaseExpression, cst.BaseExpression]],
) -> list[cst.SimpleStatementLine]:
    """Build ``monkeypatch.setenv(key, value)`` statement lines."""

    out: list[cst.SimpleStatementLine] = []
    for key, value in pairs:
        call = cst.Call(
            func=cst.Attribute(
                value=cst.Name("monkeypatch"),
                attr=cst.Name("setenv"),
            ),
            args=[
                cst.Arg(value=key),
                cst.Arg(value=value),
            ],
        )
        out.append(cst.SimpleStatementLine(body=[cst.Expr(value=call)]))
    return out


# ---------------------------------------------------------------------------
# Pass 1 — scan: which test functions are affected, and how
# ---------------------------------------------------------------------------


class _FunctionPlanner(cst.CSTVisitor):
    """Plan rewrites per function.

    Builds a map of FunctionDef ids → planning record:

    * ``needs_monkeypatch_param``: the function should gain a
      ``monkeypatch`` parameter.
    * ``setenv_to_prepend_for_decorators``: setenv calls that come from
      removed ``@patch.dict(...)`` decorators; should be inserted at the
      top of the function body.
    """

    METADATA_DEPENDENCIES: tuple = ()

    def __init__(self) -> None:
        super().__init__()
        self.plans: dict[int, _FunctionPlan] = {}
        self._func_stack: list[cst.FunctionDef] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self._func_stack.append(node)
        plan = _FunctionPlan(func=node)
        # Walk decorators: any @patch.dict(os.environ, …) that is
        # convertible contributes setenv lines and is dropped.
        for deco in node.decorators:
            if not isinstance(deco.decorator, cst.Call):
                continue
            call = deco.decorator
            if _is_convertible_patch_dict(call):
                pairs = _dict_literal_pairs(call)
                assert pairs is not None
                plan.decorators_to_remove.add(id(deco))
                plan.setenv_to_prepend.extend(pairs)
                plan.touched = True
            elif _is_skip_candidate_patch_dict(call):
                # Annotate so the file shows a TODO marker.
                plan.decorators_to_annotate.add(id(deco))
                plan.touched_skip = True
        self.plans[id(node)] = plan
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._func_stack.pop()

    def visit_With(self, node: cst.With) -> bool | None:
        if not self._func_stack:
            return None
        func = self._func_stack[-1]
        plan = self.plans[id(func)]
        # Examine each WithItem: convertible items become setenv lines.
        plan_with = _WithPlan(node=node)
        for item in node.items:
            if not isinstance(item.item, cst.Call):
                continue
            call = item.item
            if _is_convertible_patch_dict(call) and item.asname is None:
                pairs = _dict_literal_pairs(call)
                assert pairs is not None
                plan_with.convertible_items[id(item)] = pairs
                plan.touched = True
            elif _is_convertible_patch_dict(call) and item.asname is not None:
                # ``with patch.dict("os.environ", {...}) as env:`` —
                # the bound name is referenced inside the body, so we
                # cannot convert to statement-based ``setenv``.
                plan_with.skip_items.add(id(item))
                plan.touched_skip = True
            elif _is_skip_candidate_patch_dict(call):
                plan_with.skip_items.add(id(item))
                plan.touched_skip = True
        if plan_with.convertible_items or plan_with.skip_items:
            plan.with_plans[id(node)] = plan_with
        return None


class _WithPlan:
    """Per-``With`` planning record."""

    def __init__(self, node: cst.With) -> None:
        self.node = node
        # WithItem id -> setenv pairs to emit
        self.convertible_items: dict[
            int,
            list[tuple[cst.BaseExpression, cst.BaseExpression]],
        ] = {}
        # WithItem ids that should receive a TODO comment
        self.skip_items: set[int] = set()


class _FunctionPlan:
    """Per-FunctionDef planning record."""

    def __init__(self, func: cst.FunctionDef) -> None:
        self.func = func
        self.touched = False
        self.touched_skip = False
        self.decorators_to_remove: set[int] = set()
        self.decorators_to_annotate: set[int] = set()
        self.setenv_to_prepend: list[
            tuple[cst.BaseExpression, cst.BaseExpression]
        ] = []
        self.with_plans: dict[int, _WithPlan] = {}


# ---------------------------------------------------------------------------
# Pass 2 — rewrite
# ---------------------------------------------------------------------------


def _function_has_monkeypatch_param(func: cst.FunctionDef) -> bool:
    """Check whether the function already binds a ``monkeypatch`` param."""

    params = func.params
    for p in (*params.params, *params.posonly_params, *params.kwonly_params):
        if isinstance(p.name, cst.Name) and p.name.value == "monkeypatch":
            return True
    if params.star_arg is not None and isinstance(params.star_arg, cst.Param):
        if (
            isinstance(params.star_arg.name, cst.Name)
            and params.star_arg.name.value == "monkeypatch"
        ):
            return True
    if params.star_kwarg is not None:
        if (
            isinstance(params.star_kwarg.name, cst.Name)
            and params.star_kwarg.name.value == "monkeypatch"
        ):
            return True
    return False


def _add_monkeypatch_param(params: cst.Parameters) -> cst.Parameters:
    """Return a copy of ``params`` with ``monkeypatch`` appended.

    Inserts at the end of positional params (before *args / **kwargs).
    """

    new_param = cst.Param(name=cst.Name("monkeypatch"))
    new_positional = list(params.params)
    # If there's an existing positional, give the prior last param a comma.
    if new_positional:
        last = new_positional[-1]
        new_positional[-1] = last.with_changes(
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
    new_positional.append(new_param)
    return params.with_changes(params=new_positional)


def _annotate_with_todo(stmt: cst.With) -> cst.With:
    """Prepend ``# TODO(codemod)`` to a With statement's leading_lines.

    Idempotent — if the comment is already there, no change.
    """

    existing = list(stmt.leading_lines)
    for line in existing:
        if (
            isinstance(line, cst.EmptyLine)
            and line.comment is not None
            and _TODO_COMMENT in line.comment.value
        ):
            return stmt
    todo = cst.EmptyLine(
        indent=True,
        comment=cst.Comment(_TODO_COMMENT),
    )
    return stmt.with_changes(leading_lines=[*existing, todo])


def _annotate_decorator_todo(deco: cst.Decorator) -> cst.Decorator:
    """Prepend ``# TODO(codemod)`` to a decorator's leading_lines.

    Idempotent.
    """

    existing = list(deco.leading_lines)
    for line in existing:
        if (
            isinstance(line, cst.EmptyLine)
            and line.comment is not None
            and _TODO_COMMENT in line.comment.value
        ):
            return deco
    todo = cst.EmptyLine(
        indent=True,
        comment=cst.Comment(_TODO_COMMENT),
    )
    return deco.with_changes(leading_lines=[*existing, todo])


class _Rewriter(cst.CSTTransformer):
    """Apply the planned rewrites."""

    def __init__(self, plans: dict[int, _FunctionPlan]) -> None:
        super().__init__()
        self._plans = plans
        self.changed_count = 0
        self.skip_count = 0
        # Decorator and with-item ids → indexed by their owning function.
        self._deco_to_remove: set[int] = set()
        self._deco_to_annotate: set[int] = set()
        self._with_plans: dict[int, _WithPlan] = {}
        for plan in plans.values():
            self._deco_to_remove |= plan.decorators_to_remove
            self._deco_to_annotate |= plan.decorators_to_annotate
            for k, v in plan.with_plans.items():
                self._with_plans[k] = v

    # ---- decorator handling --------------------------------------------

    def leave_Decorator(
        self,
        original_node: cst.Decorator,
        updated_node: cst.Decorator,
    ) -> cst.Decorator | cst.RemovalSentinel:
        if id(original_node) in self._deco_to_remove:
            self.changed_count += 1
            return cst.RemoveFromParent()
        if id(original_node) in self._deco_to_annotate:
            self.skip_count += 1
            return _annotate_decorator_todo(updated_node)
        return updated_node

    # ---- With handling -------------------------------------------------

    def leave_With(
        self,
        original_node: cst.With,
        updated_node: cst.With,
    ) -> cst.BaseStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        plan = self._with_plans.get(id(original_node))
        if plan is None:
            return updated_node

        # Build new item list: drop convertible items entirely.
        new_items: list[cst.WithItem] = []
        emitted_setenvs: list[cst.SimpleStatementLine] = []
        skipped_any = False
        for orig_item, upd_item in zip(
            original_node.items, updated_node.items, strict=True,
        ):
            if id(orig_item) in plan.convertible_items:
                pairs = plan.convertible_items[id(orig_item)]
                emitted_setenvs.extend(_build_setenv_calls(pairs))
                self.changed_count += 1
                continue
            if id(orig_item) in plan.skip_items:
                skipped_any = True
            new_items.append(upd_item)

        # If no items remain, dissolve the With into:
        #   setenv lines + body statements (dedented).
        if not new_items:
            # The body is an IndentedBlock; flatten its statements.
            body = updated_node.body
            stmts: list[cst.BaseStatement] = []
            stmts.extend(emitted_setenvs)
            assert isinstance(body, cst.IndentedBlock)
            # Preserve the leading_lines (comments before the with) by
            # attaching them to the first emitted statement.
            if emitted_setenvs and updated_node.leading_lines:
                first = emitted_setenvs[0]
                if isinstance(first, cst.SimpleStatementLine):
                    stmts[0] = first.with_changes(
                        leading_lines=list(updated_node.leading_lines),
                    )
            stmts.extend(body.body)
            return cst.FlattenSentinel(stmts)

        # Otherwise: prepend setenv lines BEFORE the modified With and
        # keep the rest of the items.  When we drop items from a
        # compound ``with (A, B):`` we must also normalize the last
        # surviving item's trailing comma; libcst's ``Comma`` on the
        # prior item still references the dropped item's layout, which
        # leaves a stray blank line with whitespace (ruff W293).
        new_items = _strip_trailing_comma(new_items)
        new_with = updated_node.with_changes(items=new_items)
        if skipped_any:
            new_with = _annotate_with_todo(new_with)
            self.skip_count += 1
        if emitted_setenvs:
            # Attach the original leading_lines to the first setenv.
            if updated_node.leading_lines:
                first = emitted_setenvs[0]
                if isinstance(first, cst.SimpleStatementLine):
                    emitted_setenvs[0] = first.with_changes(
                        leading_lines=list(updated_node.leading_lines),
                    )
                new_with = new_with.with_changes(leading_lines=[])
            return cst.FlattenSentinel([*emitted_setenvs, new_with])

        # No conversions; only annotation.
        return new_with

    # ---- FunctionDef handling -----------------------------------------

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.BaseStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        plan = self._plans.get(id(original_node))
        if plan is None or not (plan.touched or plan.touched_skip):
            return updated_node

        # Add monkeypatch param if the function was touched (i.e. we
        # actually introduce a monkeypatch.* call somewhere) and it
        # doesn't already have one.
        new_func = updated_node
        if plan.touched and not _function_has_monkeypatch_param(updated_node):
            new_func = new_func.with_changes(
                params=_add_monkeypatch_param(updated_node.params),
            )

        # Prepend setenv lines from removed decorators.
        if plan.setenv_to_prepend:
            preamble = _build_setenv_calls(plan.setenv_to_prepend)
            body = new_func.body
            assert isinstance(body, cst.IndentedBlock)
            new_body = body.with_changes(
                body=[*preamble, *body.body],
            )
            new_func = new_func.with_changes(body=new_body)
        return new_func


# ---------------------------------------------------------------------------
# Import management
# ---------------------------------------------------------------------------


class _PatchUsageScanner(cst.CSTVisitor):
    """Detect whether ``patch`` (the name) is still referenced.

    We treat any non-import ``patch`` reference as "still used".
    """

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
        if node.value == "patch":
            self.found = True


class _ImportFixer(cst.CSTTransformer):
    """Drop ``patch`` from ``unittest.mock`` imports if no longer used."""

    def __init__(self, *, patch_still_used: bool) -> None:
        super().__init__()
        self._patch_still_used = patch_still_used

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        if self._patch_still_used:
            return updated_node
        if not (
            isinstance(updated_node.module, cst.Attribute | cst.Name)
            and _module_dotted_name(updated_node.module) == "unittest.mock"
        ):
            return updated_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        kept: list[cst.ImportAlias] = []
        for alias in updated_node.names:
            assert isinstance(alias, cst.ImportAlias)
            if (
                isinstance(alias.name, cst.Name)
                and alias.name.value == "patch"
                and alias.asname is None
            ):
                continue
            kept.append(alias)
        if not kept:
            return cst.RemoveFromParent()
        kept = _normalize_alias_commas(kept)
        return updated_node.with_changes(names=kept)


def _module_dotted_name(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_module_dotted_name(node.value)}.{node.attr.value}"
    return ""


def _normalize_alias_commas(
    aliases: list[cst.ImportAlias],
) -> list[cst.ImportAlias]:
    """Ensure all but the last alias has a comma+space."""

    out: list[cst.ImportAlias] = []
    last = len(aliases) - 1
    for i, alias in enumerate(aliases):
        if i < last:
            out.append(
                alias.with_changes(
                    comma=cst.Comma(
                        whitespace_after=cst.SimpleWhitespace(" "),
                    ),
                ),
            )
        else:
            out.append(alias.with_changes(comma=cst.MaybeSentinel.DEFAULT))
    return out


# ---------------------------------------------------------------------------
# Top-level Codemod
# ---------------------------------------------------------------------------


class PatchDictEnvironToMonkeypatchCodemod(Codemod):
    """Replace ``patch.dict("os.environ", {...})`` with monkeypatch.setenv.

    See module docstring for full conversion rules and skipped forms.
    """

    DESCRIPTION = (
        "Replace patch.dict('os.environ', {...}) with monkeypatch.setenv(...)"
    )

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)
        self.changed_count: int = 0
        self.skip_count: int = 0

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        # Pass 1 — plan per-function.
        planner = _FunctionPlanner()
        tree.visit(planner)

        # If nothing convertible AND nothing to annotate, bail.
        if not any(p.touched or p.touched_skip for p in planner.plans.values()):
            return tree

        # Pass 2 — rewrite.
        rewriter = _Rewriter(planner.plans)
        tree = tree.visit(rewriter)
        self.changed_count = rewriter.changed_count
        self.skip_count = rewriter.skip_count

        # Pass 3 — drop ``from unittest.mock import patch`` if unused.
        usage = _PatchUsageScanner()
        tree.visit(usage)
        fixer = _ImportFixer(patch_still_used=usage.found)
        tree = tree.visit(fixer)
        return tree


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


def _run_on_file(path: Path, *, apply: bool) -> tuple[int, int, str]:
    """Run the codemod on a single file.

    Return ``(changed_count, skip_count, diff)``.
    """

    src = path.read_text()
    context = CodemodContext(filename=str(path))
    codemod = PatchDictEnvironToMonkeypatchCodemod(context)
    tree = cst.parse_module(src)
    new_tree = codemod.transform_module_impl(tree)
    new_src = new_tree.code
    if new_src == src:
        return 0, 0, ""
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
    return codemod.changed_count, codemod.skip_count, diff


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.codemods.patchdict_environ_to_monkeypatch",
        description=PatchDictEnvironToMonkeypatchCodemod.DESCRIPTION,
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
        help="Print unified diffs but do not write (default).",
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
    total_skips = 0
    for path in _iter_target_files(args.paths):
        total_files += 1
        try:
            n, skip, diff = _run_on_file(path, apply=apply)
        except cst.ParserSyntaxError as e:
            print(f"SKIP {path}: parse error: {e}", file=sys.stderr)
            continue
        if n == 0 and skip == 0:
            continue
        changed_files += 1
        total_changes += n
        total_skips += skip
        if not args.quiet:
            sys.stdout.write(diff)
    sys.stdout.write(
        f"\n[patchdict_environ_to_monkeypatch] "
        f"scanned {total_files} files, "
        f"{'rewrote' if apply else 'would rewrite'} {changed_files} files "
        f"({total_changes} conversions, {total_skips} skip-annotations).\n",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
