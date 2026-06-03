# Saved Models

Store trained inference artifacts here during development.

Recommended formats:
- `joblib` / `.pkl` for scikit-learn style models and other lightweight Python objects
- `.pt` / `.pth` for PyTorch models
- `onnx` if you want a deployment-friendly portable format

For the demo, Flink should load the model once per task and reuse it for all records.
