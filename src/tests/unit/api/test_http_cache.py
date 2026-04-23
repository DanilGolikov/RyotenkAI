"""Tests for the HTTP conditional-GET helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
from fastapi import Response
from starlette.datastructures import Headers
from starlette.requests import Request

from src.api.http_cache import apply_cache_headers, etag_for, is_fresh, last_modified_for


def _make_request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": Headers(headers).raw,
    }
    return Request(scope)


def test_etag_is_weak_and_contains_mtime() -> None:
    tag = etag_for(123456789)
    assert tag.startswith("W/")
    assert "123456789" in tag


def test_last_modified_is_rfc7231_gmt() -> None:
    # 2024-01-02 03:04:05 UTC in nanoseconds
    dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    mtime_ns = int(dt.timestamp()) * 1_000_000_000
    value = last_modified_for(mtime_ns)
    assert value.endswith("GMT")
    assert "02 Jan 2024 03:04:05" in value


def test_is_fresh_matches_on_if_none_match() -> None:
    mtime = 42
    request = _make_request({"if-none-match": etag_for(mtime)})
    assert is_fresh(request, mtime) is True


def test_is_fresh_handles_comma_separated_list() -> None:
    request = _make_request({"if-none-match": f'W/"1", {etag_for(42)}, W/"99"'})
    assert is_fresh(request, 42) is True


def test_is_fresh_mismatch_on_if_none_match_ignores_if_modified_since() -> None:
    # RFC 7232 §6: If-None-Match takes precedence. A stale ETag means the
    # client's cache is stale — even if their If-Modified-Since would otherwise
    # match, we still return the full body.
    mtime = 42
    past = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=10_000)
    request = _make_request(
        {
            "if-none-match": 'W/"stale"',
            "if-modified-since": format_datetime(past, usegmt=True),
        }
    )
    assert is_fresh(request, mtime) is False


def test_is_fresh_uses_if_modified_since_when_no_etag() -> None:
    dt = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
    mtime_ns = int(dt.timestamp()) * 1_000_000_000
    # Client says "I already have a copy from this exact second or later".
    request = _make_request({"if-modified-since": format_datetime(dt, usegmt=True)})
    assert is_fresh(request, mtime_ns) is True


def test_is_fresh_returns_false_when_client_older_than_server() -> None:
    server_dt = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)
    client_dt = server_dt - timedelta(seconds=5)
    mtime_ns = int(server_dt.timestamp()) * 1_000_000_000
    request = _make_request({"if-modified-since": format_datetime(client_dt, usegmt=True)})
    assert is_fresh(request, mtime_ns) is False


def test_is_fresh_without_validators_is_false() -> None:
    assert is_fresh(_make_request({}), 42) is False


def test_apply_cache_headers_sets_all_three() -> None:
    response = Response()
    apply_cache_headers(response, 123)
    assert response.headers["ETag"] == etag_for(123)
    assert "GMT" in response.headers["Last-Modified"]
    assert response.headers["Cache-Control"] == "no-cache"


@pytest.mark.parametrize("bad_header", ["", "not a date", "Fri, 32 Foo 2024"])
def test_is_fresh_tolerates_malformed_if_modified_since(bad_header: str) -> None:
    request = _make_request({"if-modified-since": bad_header})
    assert is_fresh(request, 1) is False
