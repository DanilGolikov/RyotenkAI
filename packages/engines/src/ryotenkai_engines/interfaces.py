"""``IInferenceEngine`` Protocol — the engine plugin contract.

PR-1 stub — landed in PR-2. The shape:

    @runtime_checkable
    class IInferenceEngine(Protocol):
        engine_id:    ClassVar[str]
        config_class: ClassVar[type[BaseEngineConfig]]

        def get_capabilities(self) -> EngineCapabilities: ...
        def build_launch_spec(self, *, cfg, image, container_name, port,
                              workspace_host_path, model_path_in_container) -> LaunchSpec: ...
        def build_healthcheck_command(self, *, host: str, port: int) -> str: ...
        def build_default_endpoint_url(self, *, host: str, port: int) -> str: ...
        def validate_config(self, cfg) -> Result[None, AppError]: ...

Plus the supporting types (``BaseEngineConfig``, ``LaunchSpec``).

The Protocol is **side-effect free** — methods only return data. Provider
classes own the actual launch (``docker run`` / k8s Pod / systemd unit /
RunPod API).
"""

from __future__ import annotations

# TODO(PR-2): implement IInferenceEngine, BaseEngineConfig, LaunchSpec.
__all__: tuple[str, ...] = ()
