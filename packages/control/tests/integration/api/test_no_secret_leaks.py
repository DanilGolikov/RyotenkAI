"""Invariant: no response schema exposes a plaintext ``token`` field.

The only place ``token`` may appear in the OpenAPI spec is the *request*
body of PUT /providers/{id}/token and PUT /integrations/{id}/token. If
a future change introduces a response field named ``token``, this test
fails loudly before the change ships.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _walk(node: object, path: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield f"{path}.{k}", v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield f"{path}[{i}]", v
            yield from _walk(v, f"{path}[{i}]")


def test_no_response_schema_exposes_token(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()

    for route_path, operations in spec.get("paths", {}).items():
        if not isinstance(operations, dict):
            continue
        for method, op in operations.items():
            if method in {"parameters", "summary", "description"}:
                continue
            if not isinstance(op, dict):
                continue
            responses = op.get("responses") or {}
            for status_code, resp_obj in responses.items():
                content = (resp_obj or {}).get("content") or {}
                for media_obj in content.values():
                    schema = (media_obj or {}).get("schema")
                    if not schema:
                        continue
                    # Inline schema OR $ref. Resolve $ref within this spec.
                    resolved = _resolve_schema(spec, schema)
                    _assert_no_token_property(
                        resolved,
                        where=f"{method.upper()} {route_path} → {status_code}",
                        spec=spec,
                    )


def _resolve_schema(spec: dict, schema: dict, seen: set[str] | None = None) -> dict:
    seen = seen or set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            return {}
        seen.add(ref)
        # refs look like "#/components/schemas/Foo"
        segments = ref.lstrip("#/").split("/")
        cursor: object = spec
        for seg in segments:
            if isinstance(cursor, dict):
                cursor = cursor.get(seg, {})
        if not isinstance(cursor, dict):
            return {}
        schema = cursor
    return schema


def _assert_no_token_property(schema: dict, *, where: str, spec: dict) -> None:
    if not isinstance(schema, dict):
        return
    props = schema.get("properties") or {}
    assert (
        "token" not in props
    ), f"response schema at {where} leaks a plaintext token property — check {json.dumps(schema)[:300]}"
    # Recurse into nested object properties and array items.
    for value in props.values():
        if isinstance(value, dict):
            inner = _resolve_schema(spec, value)
            _assert_no_token_property(inner, where=where, spec=spec)
    items = schema.get("items")
    if isinstance(items, dict):
        inner = _resolve_schema(spec, items)
        _assert_no_token_property(inner, where=where, spec=spec)
    for key in ("oneOf", "anyOf", "allOf"):
        for branch in schema.get(key, []) or []:
            if isinstance(branch, dict):
                inner = _resolve_schema(spec, branch)
                _assert_no_token_property(inner, where=where, spec=spec)
