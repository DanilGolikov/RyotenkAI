from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.validators.providers import validate_ssh
from src.utils.config import SSHConfig

pytestmark = pytest.mark.unit


class TestSSHConfigValidators:
    def test_positive_alias_mode(self) -> None:
        cfg = SSHConfig(alias="pc")
        validate_ssh(cfg)

    def test_positive_explicit_mode(self) -> None:
        cfg = SSHConfig(host="127.0.0.1", user="root")
        validate_ssh(cfg)

    def test_invariant_alias_wins_even_if_host_user_missing(self) -> None:
        cfg = SSHConfig(alias="pc", host=None, user=None)
        validate_ssh(cfg)

    def test_invariant_alias_allows_host_user_too(self) -> None:
        cfg = SSHConfig(alias="pc", host="127.0.0.1", user="root")
        validate_ssh(cfg)

    def test_negative_missing_host_and_user(self) -> None:
        with pytest.raises(ValidationError, match=r"SSHConfig requires either ssh\.alias or"):
            _ = SSHConfig()

    def test_boundary_empty_alias_treated_as_missing(self) -> None:
        with pytest.raises(ValidationError, match=r"SSHConfig requires either ssh\.alias or"):
            _ = SSHConfig(alias="")

