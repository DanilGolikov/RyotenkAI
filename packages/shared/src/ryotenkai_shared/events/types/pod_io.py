"""Pod-domain trainer subprocess IO events.

Two event types — line-buffered stdout and stderr from the trainer
subprocess. Producer: runner subprocess reader threads. Gated behind a
debug-only flag in production because volume is high.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent

# Type-level discriminator for the source stream — keeps the union closed
# at one place while preserving "which fd" information in the payload.
IOStream = Literal["stdout", "stderr"]


class TrainerStdoutPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    line: str
    stream: Literal["stdout"] = "stdout"


class TrainerStdoutEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.io.trainer_stdout"] = (
        "ryotenkai.pod.io.trainer_stdout"
    )
    severity: Literal["debug"] = "debug"
    payload: TrainerStdoutPayload


class TrainerStderrPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    line: str
    stream: Literal["stderr"] = "stderr"


class TrainerStderrEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.io.trainer_stderr"] = (
        "ryotenkai.pod.io.trainer_stderr"
    )
    severity: Literal["debug"] = "debug"
    payload: TrainerStderrPayload


__all__ = [
    "IOStream",
    "TrainerStderrEvent",
    "TrainerStderrPayload",
    "TrainerStdoutEvent",
    "TrainerStdoutPayload",
]
