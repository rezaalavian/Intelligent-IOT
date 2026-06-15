# Run the Intelligent-IOT Streamlit dashboard (Windows-friendly launcher).
# Usage: powershell -File scripts/run_dashboard.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = "C:\Users\73rez\miniconda3\envs\Intelligent-IOT-blackwell\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Intelligent-IOT-blackwell env not found at $Python"
    Write-Host "Activate your env and run:"
    Write-Host "  python -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py"
    exit 1
}

Write-Host "Starting dashboard from $RepoRoot"
Write-Host "Open: http://localhost:8501"
& $Python -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py --server.port 8501
