# Environment Setup

Create the GPU-ready Conda environment:

```powershell
conda env create -f environment.yml
conda activate Intelligent-IOT
```

If you already created a previous environment named `Realtime-IOT`, remove it first to avoid confusion:

```powershell
conda env remove -n Realtime-IOT
```

Notes:
- The environment is pinned for Python 3.10 and CUDA 12.1-compatible PyTorch packages.
- Apache Flink is not installed by Conda/Pip here; use a local Flink distribution or the offline Phase 2 stubs in `analytics/flink_jobs/` for the demo.
- If your NVIDIA driver is older than CUDA 12.1 support, tell me and I will switch the env to a compatible CUDA build.
