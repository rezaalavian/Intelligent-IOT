Development environment notes

Prerequisites
- Docker & Docker Compose
- Conda (optional) or Python 3.10
- (Optional) Tesseract for OCR if extracting scanned PDFs

Create conda env:
```powershell
conda env create -f environment.yml
conda activate Intelligent-IOT
pip install -r requirements.txt
```

Current project environment name: `Intelligent-IOT`.

Start infra (PowerShell):
```powershell
.\scripts\dev_up.ps1
```
