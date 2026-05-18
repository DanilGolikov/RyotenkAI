"""Slice builders for :class:`~ryotenkai_control.reports.composer.ReportComposer`.

Each slice is a small (~80-100 LOC) focused implementation of
:class:`~ryotenkai_control.reports.composer.ReportSliceBuilder` covering
one section of the experiment report. Some slices are placeholders in
M3.A — they advertise the section but emit a clearly-labelled
``TODO`` body until the additional read-path Protocols
(``IMetricHistoryQuery``, ``IEvalResultsLoader``, etc.) land in M3.B.

Public API:

* :class:`HeaderSlice`
* :class:`MetricsSlice`
* :class:`LossCurveSlice`         — placeholder
* :class:`EvalSlice`              — placeholder
* :class:`InferenceSlice`         — placeholder
* :class:`ArtifactsSlice`
* :class:`LineageSlice`
"""

from __future__ import annotations

from ryotenkai_control.reports.slices.artifacts import ArtifactsSlice
from ryotenkai_control.reports.slices.eval import EvalSlice
from ryotenkai_control.reports.slices.header import HeaderSlice
from ryotenkai_control.reports.slices.inference import InferenceSlice
from ryotenkai_control.reports.slices.lineage import LineageSlice
from ryotenkai_control.reports.slices.loss_curve import LossCurveSlice
from ryotenkai_control.reports.slices.metrics import MetricsSlice

__all__ = [
    "ArtifactsSlice",
    "EvalSlice",
    "HeaderSlice",
    "InferenceSlice",
    "LineageSlice",
    "LossCurveSlice",
    "MetricsSlice",
]
