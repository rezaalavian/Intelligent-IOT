"""Flink-friendly inference stub.

This module keeps prediction CPU-based and loads the model artifact once.
The actual Flink job can call `load_predictor` in an open() hook or similar setup step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from models.model_io import load_model


def load_predictor(model_path: str | Path) -> Any:
    """Load the serialized model artifact for repeated CPU inference."""

    return load_model(model_path)


def predict_record(predictor: Any, features: dict[str, Any]) -> Any:
    """Predict a single record using a preloaded model.

    The predictor is expected to expose either `predict` or `predict_proba`.
    """

    if hasattr(predictor, "predict"):
        row = _to_feature_row(features)
        return predictor.predict([row])
    if callable(predictor):
        return predictor(features)
    raise TypeError("Unsupported predictor type")


def _to_feature_row(features: Mapping[str, Any] | Sequence[Any]) -> list[Any]:
    """Convert a feature mapping into a flat numeric row for CPU inference.

    Dict keys are sorted so the same feature order is used at training and inference
    time as long as the calling code keeps the schema stable.
    """

    if isinstance(features, Mapping):
        return [features[key] for key in sorted(features.keys())]
    return list(features)
