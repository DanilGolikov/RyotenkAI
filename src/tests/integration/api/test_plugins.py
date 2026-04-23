from __future__ import annotations

from fastapi.testclient import TestClient


def _manifest_by_id(plugins: list[dict], plugin_id: str) -> dict | None:
    for manifest in plugins:
        if manifest["id"] == plugin_id:
            return manifest
    return None


def test_list_reward_plugins_returns_seed_manifest(client: TestClient) -> None:
    resp = client.get("/api/v1/plugins/reward")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["kind"] == "reward"
    assert len(payload["plugins"]) >= 1

    seed = _manifest_by_id(payload["plugins"], "helixql_compiler_semantic")
    assert seed is not None, "seed plugin missing"
    assert seed["category"] == "semantic"
    assert seed["description"]
    assert "timeout_seconds" in seed["suggested_params"]


def test_list_validation_plugins_contains_seed(client: TestClient) -> None:
    resp = client.get("/api/v1/plugins/validation")
    assert resp.status_code == 200
    plugins = resp.json()["plugins"]
    seed = _manifest_by_id(plugins, "min_samples")
    assert seed is not None
    assert seed["suggested_thresholds"].get("threshold") == 100

    avg = _manifest_by_id(plugins, "avg_length")
    assert avg is not None
    # Manifest v3: thresholds_schema is a JSON Schema object; field names
    # live under .properties.
    assert avg["thresholds_schema"]["type"] == "object"
    assert set(avg["thresholds_schema"]["properties"].keys()) == {"min", "max"}


def test_list_evaluation_plugins_contains_seed(client: TestClient) -> None:
    resp = client.get("/api/v1/plugins/evaluation")
    assert resp.status_code == 200
    plugins = resp.json()["plugins"]
    seed = _manifest_by_id(plugins, "helixql_semantic_match")
    assert seed is not None
    assert seed["suggested_thresholds"].get("min_mean_score") == 0.7


def test_unknown_plugin_kind_is_rejected(client: TestClient) -> None:
    resp = client.get("/api/v1/plugins/bogus")
    # Literal type → FastAPI returns 422 before handler runs
    assert resp.status_code in (404, 422)


def test_baseline_plugin_has_minimal_manifest(client: TestClient) -> None:
    """Every plugin response carries the standard ui_manifest keys."""
    resp = client.get("/api/v1/plugins/validation")
    assert resp.status_code == 200
    plugins = resp.json()["plugins"]
    # pick a plugin not in our seed list
    baseline_ids = {p["id"] for p in plugins} - {"min_samples", "avg_length"}
    assert baseline_ids, "expected at least one non-seed plugin"
    sample_id = next(iter(baseline_ids))
    sample = _manifest_by_id(plugins, sample_id)
    assert sample is not None
    # Schema v3: no ``priority`` field, params/thresholds are JSON Schema objects.
    assert "priority" not in sample
    for key in ("id", "name", "version", "kind", "supported_strategies",
                "params_schema", "thresholds_schema"):
        assert key in sample


# ---------------------------------------------------------------------------
# /plugins/reports/defaults — surfaces the built-in section order so the
# frontend's Reports tab can pre-fill the list when the user config omits
# ``reports.sections``. See PluginsTab.tsx::materializeReportsIfNeeded.
# ---------------------------------------------------------------------------


def test_reports_defaults_returns_default_section_order(client: TestClient) -> None:
    resp = client.get("/api/v1/plugins/reports/defaults")
    assert resp.status_code == 200
    payload = resp.json()
    assert "sections" in payload
    sections = payload["sections"]
    assert isinstance(sections, list)
    # Non-empty — the UI relies on this to show a meaningful Reports
    # section when the user hasn't attached anything yet.
    assert len(sections) > 0
    # Known anchors from ``src/reports/plugins/defaults.py`` — first
    # section is always the header, last is always the footer. Guards
    # against accidental reorderings that would shift every user's
    # report layout on upgrade.
    assert sections[0] == "header"
    assert sections[-1] == "footer"
    # No duplicates — uniqueness invariant on the default list.
    assert len(set(sections)) == len(sections)


def test_reports_defaults_entries_are_actual_report_plugin_ids(client: TestClient) -> None:
    """Every id in the defaults list must reference a real report plugin
    in the catalog. Regression guard: shipping a default that points to
    a nonexistent plugin would crash the report generator."""
    defaults = client.get("/api/v1/plugins/reports/defaults").json()["sections"]
    catalog = client.get("/api/v1/plugins/reports").json()["plugins"]
    catalog_ids = {p["id"] for p in catalog}
    missing = [s for s in defaults if s not in catalog_ids]
    assert missing == [], f"default sections with no matching plugin: {missing}"


def test_reports_defaults_vs_plugin_list_route_order(client: TestClient) -> None:
    """The specific ``/reports/defaults`` route must NOT be shadowed by
    the generic ``/{kind}`` route. FastAPI matches routes in declaration
    order; this test would start failing the moment a refactor moves
    the generic route above the specific one and turns the endpoint
    into a 404 "unknown plugin kind 'defaults'"."""
    resp = client.get("/api/v1/plugins/reports/defaults")
    assert resp.status_code == 200
    assert "sections" in resp.json()
