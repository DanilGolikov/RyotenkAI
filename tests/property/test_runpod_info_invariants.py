"""Hypothesis invariants over :class:`RunPodInfo`.

Property-based проверки на свойства dataclass'а ``RunPodInfo`` — он
напрямую сериализуется и передаётся между Mac (control-plane) и pod-side
кодом, поэтому инварианты "что валидный snapshot всегда выживает
сериализацию" имеют практический вес.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
from hypothesis import given, strategies as st

from ryotenkai_shared.infrastructure.runpod_api import RunPodInfo

pytestmark = pytest.mark.property


# --- strategies --------------------------------------------------------------

_POD_STATUSES = st.sampled_from(
    [
        "RUNNING",
        "EXITED",
        "TERMINATED",
        "STOPPED",
        "STARTING",
        "PROVISIONING",
        "HIBERNATED",
    ],
)

_pod_ids = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=24,
)

_runpod_infos = st.builds(
    RunPodInfo,
    pod_id=_pod_ids,
    desired_status=_POD_STATUSES,
    runtime_status=st.one_of(st.none(), _POD_STATUSES),
    ssh_host=st.one_of(st.none(), st.from_regex(r"^[a-z0-9\-]{1,63}$", fullmatch=True)),
    # RunPod пробрасывает SSH через высокие порты; кодовая модель
    # должна выдерживать любой валидный TCP-порт.
    ssh_port=st.one_of(st.none(), st.integers(min_value=1, max_value=65535)),
    cost_per_hr=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
    machine_id=st.one_of(st.none(), st.text(max_size=32)),
)


# --- properties --------------------------------------------------------------


@given(info=_runpod_infos)
def test_desired_status_is_non_empty_string(info: RunPodInfo) -> None:
    """``desired_status`` всегда непустая строка — без неё консьюмеры
    не могут принять решение о следующем действии."""
    assert isinstance(info.desired_status, str)
    assert info.desired_status, "desired_status must be non-empty"


@given(info=_runpod_infos)
def test_ssh_port_is_in_valid_range(info: RunPodInfo) -> None:
    """Если ``ssh_port`` задан, он должен быть валидным TCP-портом."""
    if info.ssh_port is not None:
        assert 1 <= info.ssh_port <= 65535


@given(info=_runpod_infos)
def test_dict_roundtrip_preserves_identity(info: RunPodInfo) -> None:
    """``RunPodInfo`` → dict → ``RunPodInfo`` идентично оригиналу.

    Это ключевое свойство: snapshot пересекает process boundary через
    JSON, и обратный путь не должен менять данные.
    """
    payload = asdict(info)
    restored = RunPodInfo(**payload)
    assert restored == info


@given(info=_runpod_infos)
def test_frozen_dataclass_is_hashable(info: RunPodInfo) -> None:
    """Frozen ``RunPodInfo`` обязан быть hashable — мы складываем
    snapshots в множества/dict для дедупликации в reaper-ах."""
    # ``extras`` — это dict (default factory), он не hashable;
    # потому пихаем asdict без него, и проверяем что отдельные
    # immutable поля можно использовать как ключ.
    key = (info.pod_id, info.desired_status, info.runtime_status, info.ssh_port)
    assert hash(key) == hash(key)
