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
