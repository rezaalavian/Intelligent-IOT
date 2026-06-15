"""Model artifact helpers.

Use joblib/pickle for classical ML models and portable export formats for deep models.
The Flink side should load one artifact once per task and reuse it for all records.
"""
from pathlib import Path
from typing import Any
import joblib

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env
    torch = None


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a trained CPU inference model to disk.

    For the current project, joblib is preferred for sklearn-like models and other
    lightweight Python objects. Deep learning models should use their native format
    (for example `.pt` or ONNX) instead of joblib when possible.
    """

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if torch is not None and isinstance(model, getattr(torch.nn, "Module", tuple())):
        torch.save(model, artifact_path)
    elif artifact_path.suffix.lower() in {".pt", ".pth"} and torch is not None:
        torch.save(model, artifact_path)
    else:
        joblib.dump(model, artifact_path)
    return artifact_path


def load_model(path: str | Path) -> Any:
    """Load a persisted inference model from disk."""

    artifact_path = Path(path)
    if torch is not None and artifact_path.suffix.lower() in {".pt", ".pth"}:
        return torch.load(artifact_path, map_location="cpu")
    return joblib.load(artifact_path)
