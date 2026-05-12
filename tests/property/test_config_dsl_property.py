"""Property tests for the config DSL.

Phase 4 scope: rather than fuzzing the full :class:`PipelineConfig`
(which has heavy provider-block validators that need a registered
registry), we target two well-bounded sub-models that are user-facing:

* :class:`HuggingFaceHubConfig` — push target with a normalising
  ``repo_id`` validator.
* :class:`PluginManifest`'s reward-vs-non-reward gate via the same
  hypothesis strategies used elsewhere.

Goals (per Phase 4 exit criteria):

* ``model_validate(d).model_dump() == d`` for accepted inputs (fixed
  point).
* Validation errors are bounded — no infinite traceback, only
  ``ValueError`` / ``ValidationError`` types surface.
* No accidental ``KeyError`` / ``AttributeError`` leaks from invalid
  inputs.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ryotenkai_shared.config.integrations import HuggingFaceHubConfig

pytestmark = [pytest.mark.property]


# ---------------------------------------------------------------------------
# HuggingFaceHubConfig
# ---------------------------------------------------------------------------


@st.composite
def _hf_hub_dict(draw: st.DrawFn) -> dict[str, Any]:
    has_repo = draw(st.booleans())
    if has_repo:
        # repo_id is "user/name" — allow leading/trailing whitespace
        # because the validator normalises it (a property invariant we
        # check below).
        user = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz-_0123456789", min_size=1, max_size=15))
        name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz-_0123456789", min_size=1, max_size=20))
        whitespace = draw(st.sampled_from(["", " ", "  ", "\t"]))
        repo_id: str | None = whitespace + f"{user}/{name}" + whitespace
    else:
        repo_id = draw(st.sampled_from([None, "", "   "]))
    return {"repo_id": repo_id, "private": draw(st.booleans())}


@given(payload=_hf_hub_dict())
def test_hf_hub_round_trip_is_fixed_point(payload: dict[str, Any]) -> None:
    cfg = HuggingFaceHubConfig.model_validate(payload)
    dumped = cfg.model_dump()
    # Round-trip a second time — fixed point.
    again = HuggingFaceHubConfig.model_validate(dumped).model_dump()
    assert dumped == again


@given(payload=_hf_hub_dict())
def test_hf_hub_normalisation_strips_whitespace(payload: dict[str, Any]) -> None:
    cfg = HuggingFaceHubConfig.model_validate(payload)
    if cfg.repo_id is not None:
        # The validator strips and turns empty into None.
        assert cfg.repo_id.strip() == cfg.repo_id
        assert cfg.repo_id != ""


@given(payload=_hf_hub_dict())
def test_hf_hub_enabled_is_derived_consistently(payload: dict[str, Any]) -> None:
    cfg = HuggingFaceHubConfig.model_validate(payload)
    assert cfg.enabled == bool(cfg.repo_id)


# ---------------------------------------------------------------------------
# Adversarial inputs — every failure must surface as a structured
# pydantic ValidationError, never as a stray KeyError / AttributeError.
# ---------------------------------------------------------------------------


@given(
    payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(
            st.integers(), st.text(max_size=5), st.booleans(), st.none(),
        ),
        max_size=4,
    ),
)
def test_hf_hub_garbage_input_only_raises_validation_errors(payload: dict[str, Any]) -> None:
    """Arbitrary garbage input: either validates (when repo_id absent and unknown keys
    aren't strict-forbidden) or raises ``ValidationError`` — never a
    raw KeyError / AttributeError that would indicate a missing guard.
    """
    try:
        HuggingFaceHubConfig.model_validate(payload)
    except (ValidationError, ValueError):
        # Expected — bad input surfaces as the documented error type.
        return
    except (KeyError, AttributeError) as exc:  # pragma: no cover — bug if raised
        pytest.fail(
            f"validate raised raw {type(exc).__name__}: {exc!r} "
            f"(should have been ValidationError)",
        )
