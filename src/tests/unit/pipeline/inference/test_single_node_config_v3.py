"""
Tests for SingleNode provider config v3 structure.

Tests the new config structure:
    providers:
      single_node:
        connect:
          ssh: {...}
        training: {...}
        inference:
          serve: {...}

Coverage:
- Positive cases (valid configs)
- Negative cases (invalid/missing fields)
- Boundary cases (edge values)
- Invariants (properties that must hold)
- Dependency errors (SSH, Pydantic validation)
- Regression tests (previous bugs)
- Logic-specific tests (SSH modes, property shortcuts)
- Combinatorial tests (various config combinations)
"""

import pytest
from pydantic import ValidationError

from src.providers.single_node.training.config import (
    SingleNodeCleanupConfig,
    SingleNodeConfig,
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeTrainingConfig,
)
from src.utils.config import (
    InferenceSingleNodeServeConfig,
    SSHConfig,
    SSHConnectSettings,
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_ssh_alias():
    """Valid SSH config with alias (recommended mode)."""
    return SSHConfig(alias="pc")


@pytest.fixture
def valid_ssh_explicit():
    """Valid SSH config with explicit host+user."""
    return SSHConfig(
        host="192.168.1.100",
        port=22,
        user="testuser",
        key_path="~/.ssh/id_ed25519",
    )


@pytest.fixture
def valid_connect_config(valid_ssh_alias):
    """Valid connect config."""
    return SingleNodeConnectConfig(ssh=valid_ssh_alias)


@pytest.fixture
def valid_training_config():
    """Valid training config."""
    return SingleNodeTrainingConfig(
        workspace_path="/home/testuser/workspace",
        docker_image="test/runtime:latest",
        training_start_timeout=60,
    )


@pytest.fixture
def valid_cleanup_config():
    """Valid cleanup config."""
    return SingleNodeCleanupConfig(
        cleanup_workspace=True,
        keep_on_error=False,
        on_interrupt=True,
    )


@pytest.fixture
def valid_inference_config():
    """Valid inference config."""
    return SingleNodeInferenceConfig(
        serve=InferenceSingleNodeServeConfig(
            host="127.0.0.1",
            port=8000,
            workspace="/home/testuser/inference",
        )
    )


@pytest.fixture
def valid_full_config(valid_connect_config, valid_cleanup_config, valid_training_config, valid_inference_config):
    """Valid full SingleNodeConfig."""
    return SingleNodeConfig(
        connect=valid_connect_config,
        cleanup=valid_cleanup_config,
        training=valid_training_config,
        inference=valid_inference_config,
    )


# ============================================================================
# POSITIVE TESTS
# ============================================================================

class TestPositiveCases:
    """Test valid configurations."""

    def test_minimal_config_with_alias(self):
        """Test minimal valid config with SSH alias."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.connect.ssh.alias == "pc"
        assert config.training.workspace_path == "/workspace"
        assert config.inference.serve.port == 8000  # default

    def test_minimal_config_with_explicit_ssh(self):
        """Test minimal valid config with explicit SSH."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(host="localhost", user="test")
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.connect.ssh.host == "localhost"
        assert config.connect.ssh.user == "test"

    def test_full_config_with_all_fields(self, valid_full_config):
        """Test fully populated config."""
        config = valid_full_config

        # Connect
        assert config.connect.ssh.alias == "pc"

        # Training
        assert config.training.workspace_path == "/home/testuser/workspace"
        assert config.training.training_start_timeout == 60

        # Cleanup
        assert config.cleanup.cleanup_workspace is True
        assert config.cleanup.keep_on_error is False
        assert config.cleanup.on_interrupt is True

        # Inference
        assert config.inference.serve.host == "127.0.0.1"
        assert config.inference.serve.port == 8000
        assert config.inference.serve.workspace == "/home/testuser/inference"

    def test_config_with_custom_ssh_settings(self):
        """Test config with custom SSH connect settings."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(
                    alias="pc",
                    connect_settings=SSHConnectSettings(
                        max_retries=5,
                        retry_delay_seconds=20,
                        timeout_seconds=300,
                    )
                )
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.connect.ssh.connect_settings.max_retries == 5
        assert config.connect.ssh.connect_settings.retry_delay_seconds == 20
        assert config.connect.ssh.connect_settings.timeout_seconds == 300


# ============================================================================
# NEGATIVE TESTS
# ============================================================================

class TestNegativeCases:
    """Test invalid configurations."""

    def test_missing_connect_field(self):
        """Test that missing 'connect' field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SingleNodeConfig(
                # connect missing!
                training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
                inference=SingleNodeInferenceConfig(),
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("connect",) and e["type"] == "missing" for e in errors)

    def test_missing_training_field(self):
        """Test that missing 'training' field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SingleNodeConfig(
                connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
                # training missing!
                inference=SingleNodeInferenceConfig(),
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("training",) and e["type"] == "missing" for e in errors)

    def test_invalid_workspace_path_relative(self):
        """Test that relative workspace_path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SingleNodeTrainingConfig(workspace_path="relative/path", docker_image="test/runtime:latest")  # Must be absolute

        errors = exc_info.value.errors()
        assert any("absolute" in str(e).lower() for e in errors)

    def test_invalid_ssh_no_alias_no_host(self):
        """Test that SSH config without alias or host+user is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SSHConfig()  # No alias, no host, no user

        errors = exc_info.value.errors()
        # Should fail validation: requires either alias or (host + user)
        assert len(errors) > 0

    def test_invalid_ssh_host_without_user(self):
        """Test that SSH config with host but no user is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SSHConfig(host="localhost")  # host without user

        errors = exc_info.value.errors()
        assert any("user" in str(e).lower() or "alias" in str(e).lower() for e in errors)

    def test_invalid_port_negative(self):
        """Test that negative port is rejected."""
        with pytest.raises(ValidationError):
            SSHConfig(host="localhost", user="test", port=-1)

    def test_invalid_port_too_large(self):
        """Test that port > 65535 is rejected."""
        with pytest.raises(ValidationError):
            SSHConfig(host="localhost", user="test", port=70000)

    def test_invalid_training_timeout_too_small(self):
        """Test that training_start_timeout < 10 is rejected."""
        with pytest.raises(ValidationError):
            SingleNodeTrainingConfig(
                workspace_path="/workspace",
                training_start_timeout=5,  # < 10
            )

    def test_invalid_training_timeout_too_large(self):
        """Test that training_start_timeout > 600 is rejected."""
        with pytest.raises(ValidationError):
            SingleNodeTrainingConfig(
                workspace_path="/workspace",
                training_start_timeout=700,  # > 600
            )

    def test_invalid_inference_workspace_relative(self):
        """Test that relative inference workspace is rejected."""
        with pytest.raises(ValidationError):
            InferenceSingleNodeServeConfig(
                workspace="relative/path",  # Must be absolute
            )


# ============================================================================
# BOUNDARY TESTS
# ============================================================================

class TestBoundaryCases:
    """Test edge values."""

    def test_min_valid_port(self):
        """Test minimum valid port (1)."""
        config = SSHConfig(host="localhost", user="test", port=1)
        assert config.port == 1

    def test_max_valid_port(self):
        """Test maximum valid port (65535)."""
        config = SSHConfig(host="localhost", user="test", port=65535)
        assert config.port == 65535

    def test_min_valid_training_timeout(self):
        """Test minimum valid training_start_timeout (10)."""
        config = SingleNodeTrainingConfig(
            workspace_path="/workspace",
            docker_image="test/runtime:latest",
            training_start_timeout=10,
        )
        assert config.training_start_timeout == 10

    def test_max_valid_training_timeout(self):
        """Test maximum valid training_start_timeout (600)."""
        config = SingleNodeTrainingConfig(
            workspace_path="/workspace",
            docker_image="test/runtime:latest",
            training_start_timeout=600,
        )
        assert config.training_start_timeout == 600

    def test_min_valid_ssh_retries(self):
        """Test minimum valid max_retries (1)."""
        config = SSHConfig(
            alias="pc",
            connect_settings=SSHConnectSettings(max_retries=1),
        )
        assert config.connect_settings.max_retries == 1

    def test_max_valid_ssh_retries(self):
        """Test maximum valid max_retries (10)."""
        config = SSHConfig(
            alias="pc",
            connect_settings=SSHConnectSettings(max_retries=10),
        )
        assert config.connect_settings.max_retries == 10


# ============================================================================
# INVARIANT TESTS
# ============================================================================

class TestInvariants:
    """Test properties that must always hold."""

    def test_workspace_path_always_absolute(self, valid_full_config):
        """Test that workspace_path is always absolute."""
        config = valid_full_config
        assert config.training.workspace_path.startswith("/")

    def test_inference_workspace_always_absolute(self, valid_full_config):
        """Test that inference workspace is always absolute."""
        config = valid_full_config
        assert config.inference.serve.workspace.startswith("/")

    def test_ssh_alias_or_explicit_mode(self, valid_full_config):
        """Test that SSH is either alias mode or explicit mode."""
        config = valid_full_config
        ssh = config.connect.ssh

        # Must have either alias or (host + user)
        assert ssh.alias is not None or (ssh.host is not None and ssh.user is not None)

    def test_convenience_properties_match_nested_values(self, valid_full_config):
        """Test that convenience properties return correct nested values."""
        config = valid_full_config

        # SSH shortcut
        assert config.ssh == config.connect.ssh

        # Training shortcuts
        assert config.workspace_path == config.training.workspace_path
        assert config.cleanup_workspace == config.cleanup.cleanup_workspace
        assert config.keep_on_error == config.cleanup.keep_on_error
        assert config.training_start_timeout == config.training.training_start_timeout
        assert config.gpu_type == config.training.gpu_type
        assert config.mock_mode == config.training.mock_mode

    def test_is_alias_mode_invariant(self):
        """Test that is_alias_mode matches ssh.alias presence."""
        # Alias mode
        config_alias = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )
        assert config_alias.is_alias_mode is True
        assert config_alias.connect.ssh.alias is not None

        # Explicit mode
        config_explicit = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(host="localhost", user="test")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )
        assert config_explicit.is_alias_mode is False
        assert config_explicit.connect.ssh.alias is None


# ============================================================================
# DEPENDENCY ERROR TESTS
# ============================================================================

class TestDependencyErrors:
    """Test handling of dependency-related errors."""

    def test_ssh_key_path_not_found(self):
        """Test that non-existent SSH key path raises error when resolved."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(
                    host="localhost",
                    user="test",
                    key_path="/nonexistent/key",
                )
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        # Should raise error when trying to resolve key path
        with pytest.raises(ValueError, match="SSH key not found"):
            config.resolve_ssh_key_path_for_client()

    def test_ssh_key_env_not_set(self):
        """Test that missing env var for SSH key raises error."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(
                    host="localhost",
                    user="test",
                    key_env="NONEXISTENT_ENV_VAR",
                )
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        # Should raise error when env var not set
        with pytest.raises(ValueError, match="not set"):
            config.resolve_ssh_key_path_for_client()


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Test for previously fixed bugs."""

    def test_ssh_not_duplicated_v3(self):
        """REGRESSION: v2 had SSH duplicated in providers and inference.providers."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        # v3: SSH is ONLY in connect.ssh (single source of truth)
        assert hasattr(config, "connect")
        assert hasattr(config.connect, "ssh")

        # v3: inference does NOT have ssh field
        assert not hasattr(config.inference, "ssh")
        assert hasattr(config.inference, "serve")  # Only serve config

    def test_model_dump_includes_all_sections(self, valid_full_config):
        """REGRESSION: Ensure model_dump() includes all sections."""
        config = valid_full_config
        dumped = config.model_dump(mode="python")

        assert "connect" in dumped
        assert "training" in dumped
        assert "inference" in dumped
        assert "ssh" in dumped["connect"]
        assert "workspace_path" in dumped["training"]
        assert "serve" in dumped["inference"]


# ============================================================================
# LOGIC-SPECIFIC TESTS
# ============================================================================

class TestLogicSpecific:
    """Test specific logic implementations."""

    def test_ssh_mode_detection_alias(self):
        """Test SSH mode detection for alias mode."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.is_alias_mode is True
        assert config.get_ssh_host_for_client() == "pc"
        assert config.get_ssh_user_for_client() is None  # Alias mode returns None

    def test_ssh_mode_detection_explicit(self):
        """Test SSH mode detection for explicit mode."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(host="192.168.1.100", user="testuser", port=2222)
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.is_alias_mode is False
        assert config.get_ssh_host_for_client() == "192.168.1.100"
        assert config.get_ssh_user_for_client() == "testuser"
        assert config.get_ssh_port_for_client() == 2222

    def test_ssh_key_resolution_precedence(self, tmp_path):
        """Test SSH key resolution: key_path > key_env > None."""
        # Create temp key file
        key_file = tmp_path / "test_key"
        key_file.write_text("fake key")

        # key_path takes precedence
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(
                ssh=SSHConfig(
                    host="localhost",
                    user="test",
                    key_path=str(key_file),
                    key_env="SOME_ENV",  # Should be ignored
                )
            ),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        resolved = config.resolve_ssh_key_path_for_client()
        assert resolved == str(key_file)

    def test_ssh_key_resolution_alias_mode_returns_none(self):
        """Test that alias mode returns None for key path (handled by ~/.ssh/config)."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(),
        )

        resolved = config.resolve_ssh_key_path_for_client()
        assert resolved is None


# ============================================================================
# COMBINATORIAL TESTS
# ============================================================================

class TestCombinatorial:
    """Test various combinations of config options."""

    @pytest.mark.parametrize("cleanup,keep_on_error", [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ])
    def test_lifecycle_combinations(self, cleanup, keep_on_error):
        """Test all combinations of lifecycle settings."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            cleanup=SingleNodeCleanupConfig(cleanup_workspace=cleanup, keep_on_error=keep_on_error),
            training=SingleNodeTrainingConfig(
                workspace_path="/workspace",
                docker_image="test/runtime:latest",
            ),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.cleanup.cleanup_workspace == cleanup
        assert config.cleanup.keep_on_error == keep_on_error

    @pytest.mark.parametrize("host,port", [
        ("127.0.0.1", 8000),
        ("0.0.0.0", 8000),
        ("192.168.1.100", 9000),
        ("localhost", 7860),
    ])
    def test_inference_server_combinations(self, host, port):
        """Test various inference server host/port combinations."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(workspace_path="/workspace", docker_image="test/runtime:latest"),
            inference=SingleNodeInferenceConfig(
                serve=InferenceSingleNodeServeConfig(
                    host=host,
                    port=port,
                    workspace="/inference",
                )
            ),
        )

        assert config.inference.serve.host == host
        assert config.inference.serve.port == port

    @pytest.mark.parametrize("timeout", [10, 30, 60, 120, 300, 600])
    def test_training_timeout_values(self, timeout):
        """Test various valid training timeout values."""
        config = SingleNodeConfig(
            connect=SingleNodeConnectConfig(ssh=SSHConfig(alias="pc")),
            training=SingleNodeTrainingConfig(
                workspace_path="/workspace",
                docker_image="test/runtime:latest",
                training_start_timeout=timeout,
            ),
            inference=SingleNodeInferenceConfig(),
        )

        assert config.training.training_start_timeout == timeout
