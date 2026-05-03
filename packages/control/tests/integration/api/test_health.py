from __future__ import annotations


def test_health_returns_ok_when_runs_dir_readable(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runs_dir_readable"] is True
    assert body["version"]
