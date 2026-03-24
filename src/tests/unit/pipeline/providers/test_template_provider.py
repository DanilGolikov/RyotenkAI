from __future__ import annotations

import pytest

from src.config.providers.template import TemplateProviderConfig


def test_template_provider_config_validates_type() -> None:
    cfg = TemplateProviderConfig()
    assert cfg.type == "template"

    with pytest.raises(ValueError, match="must be 'template'"):
        _ = TemplateProviderConfig(type="not-template")


def test_template_provider_config_roundtrip() -> None:
    cfg = TemplateProviderConfig.from_dict({"type": "template"})
    assert cfg.to_dict() == {"type": "template"}
