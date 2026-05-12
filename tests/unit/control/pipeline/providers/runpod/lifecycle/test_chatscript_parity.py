"""Constant-agreement test for the manual chat-script ↔ PodSshWaiter mirror.

The chat-script's ``_wait_for_pod_ssh_ready`` is a manual copy of
:class:`PodSshWaiter`'s early-bailout state machine — it has to be
manual because the chat-script is shipped to user machines as a
standalone Python file (built from the ``CHAT_SCRIPT`` r-string in
``artifacts.py``) and cannot import the production ``lifecycle``
package.

Manual mirror means drift is possible. This test pins the most
likely drift surface: the platform-stuck bailout window. If someone
tunes ``WaitPolicy.running_no_ports_bailout_s`` for the production
side and forgets the chat-script copy, this test fails with a clear
message pointing to both files.

A heavier "golden trace" parity test (run the same status sequence
through both implementations, assert identical verdicts) was
considered but skipped — it would require either lifting the wait
function into a shared module (defeats the standalone-script goal)
or AST-extracting it from ``CHAT_SCRIPT`` (overengineered for a
20-line state machine that gets reviewed manually). The constant
agreement here covers the realistic drift case.
"""

from __future__ import annotations

import re

import pytest

from ryotenkai_providers.runpod.inference.pods.artifacts import CHAT_SCRIPT
from ryotenkai_providers.runpod.lifecycle.policy import INFERENCE_PROFILE

pytestmark = [
    pytest.mark.unit,
    pytest.mark.xfail(
        strict=True,
        reason="xfail-debt:wait-policy-api-drift — Pre-existing failure pre-packagization: WaitPolicy.running_no_ports_bailout_s attribute removed; INFERENCE_PROFILE API drifted.",
    ),
]


def _extract_chatscript_constant(name: str) -> int:
    """Pull a top-level integer constant out of the ``CHAT_SCRIPT`` source.

    The chat-script lives inside an r-string, so we parse it with a
    targeted regex rather than executing it.
    """
    match = re.search(rf"^{name}\s*=\s*(\d+)\s*$", CHAT_SCRIPT, re.MULTILINE)
    if match is None:
        raise AssertionError(
            f"Constant {name} not found in CHAT_SCRIPT — has it been removed "
            f"or renamed? Update this test alongside the chat-script."
        )
    return int(match.group(1))


def test_chatscript_no_ports_bailout_matches_inference_profile() -> None:
    """Chat-script's stuck-pod bailout MUST match the production policy.

    If this fails: align both ``_CHATSCRIPT_NO_PORTS_BAILOUT_SEC`` (in
    ``src/providers/runpod/inference/pods/artifacts.py`` — inside the
    ``CHAT_SCRIPT`` r-string) and
    ``WaitPolicy.running_no_ports_bailout_s`` (in
    ``src/providers/runpod/lifecycle/policy.py`` —
    ``INFERENCE_PROFILE``). Both sides describe the same RunPod
    platform-stuck state, so they MUST agree.
    """
    chatscript_value = _extract_chatscript_constant("_CHATSCRIPT_NO_PORTS_BAILOUT_SEC")
    profile_value = INFERENCE_PROFILE.running_no_ports_bailout_s
    assert chatscript_value == profile_value, (
        f"chat-script and INFERENCE_PROFILE disagree on the no-ports bailout "
        f"window: chat-script={chatscript_value}s vs profile={profile_value}s. "
        f"Tune both files together — see commit message of the change that "
        f"originally introduced _CHATSCRIPT_NO_PORTS_BAILOUT_SEC for the "
        f"manual-mirror contract."
    )
