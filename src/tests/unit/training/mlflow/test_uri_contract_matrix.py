from __future__ import annotations

import builtins
import ssl
import sys
import urllib.error
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.integrations.mlflow import MLflowConfig
from src.infrastructure.mlflow.gateway import MLflowGateway
from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_mlflow_tracking_uri_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


def _cfg(
    *,
    tracking_uri: str | None = "https://public.example.ts.net",
    local_tracking_uri: str | None = "http://localhost:5002",
    ca_bundle_path: str | None = None,
) -> MLflowConfig:
    return MLflowConfig(
        tracking_uri=tracking_uri,
        local_tracking_uri=local_tracking_uri,
        ca_bundle_path=ca_bundle_path,
        experiment_name="test-exp",
    )


class TestPositiveAndNegative:
    def test_control_plane_prefers_local_uri_when_both_are_set(self) -> None:
        resolved = resolve_mlflow_uris(_cfg(), runtime_role="control_plane")
        assert resolved.effective_local_tracking_uri == "http://localhost:5002"
        assert resolved.effective_remote_tracking_uri == "https://public.example.ts.net"
        assert resolved.runtime_tracking_uri == "http://localhost:5002"

    def test_training_prefers_public_uri_when_both_are_set(self) -> None:
        resolved = resolve_mlflow_uris(_cfg(), runtime_role="training")
        assert resolved.effective_local_tracking_uri == "http://localhost:5002"
        assert resolved.effective_remote_tracking_uri == "https://public.example.ts.net"
        assert resolved.runtime_tracking_uri == "https://public.example.ts.net"

    def test_gateway_http_500_returns_false_with_structured_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        def fake_urlopen(req, timeout):  # noqa: ARG001
            raise urllib.error.HTTPError(url="x", code=500, msg="boom", hdrs=None, fp=None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        gateway = MLflowGateway("http://mlflow.example")
        assert gateway.check_connectivity(timeout=0.01) is False
        assert gateway.last_connectivity_error is not None
        assert gateway.last_connectivity_error.code == "MLFLOW_PREFLIGHT_HTTP_ERROR"


class TestBoundary:
    def test_whitespace_uris_are_normalized_and_trimmed(self) -> None:
        cfg = _cfg(tracking_uri="  https://public.example.ts.net  ", local_tracking_uri="  http://localhost:5002  ")
        assert cfg.tracking_uri == "https://public.example.ts.net"
        assert cfg.local_tracking_uri == "http://localhost:5002"

    def test_whitespace_env_tracking_uri_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "   ")
        resolved = resolve_mlflow_uris(_cfg(), runtime_role="training")
        assert resolved.runtime_tracking_uri == "https://public.example.ts.net"

    def test_non_http_tracking_uri_is_considered_available(self) -> None:
        gateway = MLflowGateway("file:/tmp/mlruns")
        assert gateway.check_connectivity(timeout=0.01) is True
        assert gateway.last_connectivity_error is None

    def test_https_without_custom_ca_bundle_uses_default_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        captured: dict[str, Any] = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_create_default_context(*, cafile=None, **kwargs):  # noqa: ARG001
            captured["cafile"] = cafile
            return object()

        def fake_urlopen(req, timeout, context):  # noqa: ARG001
            captured["context"] = context
            return _Resp()

        monkeypatch.setattr(ssl, "create_default_context", fake_create_default_context)
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        gateway = MLflowGateway("https://mlflow.example")
        assert gateway.check_connectivity(timeout=0.01) is True
        assert captured["cafile"] is None
        assert captured["context"] is not None


class TestInvariants:
    @pytest.mark.parametrize("runtime_role", ["control_plane", "training"])
    def test_resolver_preserves_runtime_role(self, runtime_role: str) -> None:
        resolved = resolve_mlflow_uris(_cfg(), runtime_role=runtime_role)  # type: ignore[arg-type]
        assert resolved.runtime_role == runtime_role

    @pytest.mark.parametrize(
        ("runtime_role", "expected_runtime"),
        [
            ("control_plane", "http://localhost:5002"),
            ("training", "https://public.example.ts.net"),
        ],
    )
    def test_runtime_uri_is_always_one_of_effective_uris(self, runtime_role: str, expected_runtime: str) -> None:
        resolved = resolve_mlflow_uris(_cfg(), runtime_role=runtime_role)  # type: ignore[arg-type]
        assert resolved.runtime_tracking_uri == expected_runtime
        assert resolved.runtime_tracking_uri in {
            resolved.effective_local_tracking_uri,
            resolved.effective_remote_tracking_uri,
        }

    def test_4xx_reachability_clears_last_connectivity_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        def fake_urlopen(req, timeout):  # noqa: ARG001
            raise urllib.error.HTTPError(url="x", code=404, msg="not found", hdrs=None, fp=None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        gateway = MLflowGateway("http://mlflow.example")
        assert gateway.check_connectivity(timeout=0.01) is True
        assert gateway.last_connectivity_error is None


class TestDependencyErrors:
    def test_load_prompt_import_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any):
            if name == "mlflow":
                raise ImportError("mlflow missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        gateway = MLflowGateway("http://mlflow.example")
        assert gateway.load_prompt("prompt-x", timeout=0.01) is None

    def test_load_prompt_mlflow_dependency_runtime_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mlflow = ModuleType("mlflow")
        fake_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined, ARG005]
        fake_genai = ModuleType("mlflow.genai")
        fake_genai.load_prompt = lambda name: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        fake_mlflow.genai = fake_genai  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.genai", fake_genai)

        gateway = MLflowGateway("http://mlflow.example")
        assert gateway.load_prompt("prompt-x", timeout=0.01) is None


class TestRegressions:
    def test_public_tracking_plus_localhost_does_not_force_training_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        resolved = resolve_mlflow_uris(
            _cfg(
                tracking_uri="https://macbook-air-daniil.tail43e071.ts.net",
                local_tracking_uri="http://localhost:5002",
            ),
            runtime_role="training",
        )
        assert resolved.runtime_tracking_uri == "https://macbook-air-daniil.tail43e071.ts.net"

    def test_local_only_config_still_allows_training_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        resolved = resolve_mlflow_uris(
            _cfg(tracking_uri=None, local_tracking_uri="http://localhost:5002"),
            runtime_role="training",
        )
        assert resolved.effective_local_tracking_uri == "http://localhost:5002"
        assert resolved.effective_remote_tracking_uri == "http://localhost:5002"
        assert resolved.runtime_tracking_uri == "http://localhost:5002"


class TestLogicSpecific:
    def test_explicit_env_override_affects_only_training_role(self) -> None:
        control_plane = resolve_mlflow_uris(
            _cfg(),
            runtime_role="control_plane",
            env_tracking_uri="https://override.example.ts.net",
        )
        training = resolve_mlflow_uris(
            _cfg(),
            runtime_role="training",
            env_tracking_uri="https://override.example.ts.net",
        )
        assert control_plane.runtime_tracking_uri == "http://localhost:5002"
        assert training.runtime_tracking_uri == "https://override.example.ts.net"

    def test_url_error_with_ssl_reason_maps_to_tls_code(self) -> None:
        ssl_error = ssl.SSLCertVerificationError("verify failed")
        gateway = MLflowGateway("https://mlflow.example")
        mapped = gateway._map_connectivity_error(urllib.error.URLError(ssl_error))
        assert mapped.code == "MLFLOW_TLS_CERT_VERIFY_FAILED"


class TestCombinatorial:
    @pytest.mark.parametrize(
        ("tracking_uri", "local_tracking_uri", "env_tracking_uri", "runtime_role", "expected_runtime"),
        [
            ("https://public.example.ts.net", None, None, "training", "https://public.example.ts.net"),
            (None, "http://localhost:5002", None, "training", "http://localhost:5002"),
            ("https://public.example.ts.net", "http://localhost:5002", None, "control_plane", "http://localhost:5002"),
            ("https://public.example.ts.net", "http://localhost:5002", None, "training", "https://public.example.ts.net"),
            ("https://public.example.ts.net", "http://localhost:5002", "https://override.example.ts.net", "training", "https://override.example.ts.net"),
            ("https://public.example.ts.net", "http://localhost:5002", "   ", "training", "https://public.example.ts.net"),
        ],
    )
    def test_uri_resolution_matrix(
        self,
        tracking_uri: str | None,
        local_tracking_uri: str | None,
        env_tracking_uri: str | None,
        runtime_role: str,
        expected_runtime: str,
    ) -> None:
        resolved = resolve_mlflow_uris(
            _cfg(tracking_uri=tracking_uri, local_tracking_uri=local_tracking_uri),
            runtime_role=runtime_role,  # type: ignore[arg-type]
            env_tracking_uri=env_tracking_uri,
        )
        assert resolved.runtime_tracking_uri == expected_runtime

    def test_blank_tracking_and_local_uri_rejected(self) -> None:
        with pytest.raises(ValidationError, match="At least one of 'tracking_uri' or 'local_tracking_uri' must be set"):
            _cfg(tracking_uri="   ", local_tracking_uri="   ")
