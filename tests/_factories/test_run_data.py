"""Unit tests for :mod:`tests._factories.run_data`.

These guard the factory itself: defaults yield a valid, empty ``RunData``,
and overrides are reflected on the returned dicts. Without these tests
the factory could silently break and report-test failures would point
at the consumers, not the builder.
"""

from __future__ import annotations

from mlflow.entities import RunData

from tests._factories.run_data import make_run_data


def test_defaults_produce_empty_run_data() -> None:
    rd = make_run_data()
    assert isinstance(rd, RunData)
    assert rd.metrics == {}
    assert rd.params == {}
    assert rd.tags == {}


def test_metrics_dict_is_reflected() -> None:
    rd = make_run_data(metrics={"train_loss": 0.5, "grad_norm": 1.2})
    assert rd.metrics == {"train_loss": 0.5, "grad_norm": 1.2}


def test_params_dict_is_string_coerced() -> None:
    rd = make_run_data(params={"learning_rate": 0.001, "batch_size": 32})
    # mlflow stores params as strings; the factory coerces.
    assert rd.params == {"learning_rate": "0.001", "batch_size": "32"}


def test_tags_dict_is_reflected() -> None:
    rd = make_run_data(tags={"mlflow.runName": "foo", "stage": "train"})
    assert rd.tags == {"mlflow.runName": "foo", "stage": "train"}
