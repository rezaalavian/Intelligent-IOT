param(
	[string]$EnvName = "Intelligent-IOT-blackwell",
	[switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $RepoRoot 'environment-pytorch-12.8.yml'

Write-Host "Creating conda environment: $EnvName from $EnvFile"

# check for existing env
$envs = conda env list | Out-String
if ($envs -match "\b$EnvName\b") {
	if ($Force) {
		Write-Host "Environment exists. Removing because -Force specified."
		conda env remove -n $EnvName -y
	} else {
		Write-Host "ERROR: Environment '$EnvName' already exists. Use -Force to recreate or choose a different name."
		exit 1
	}
}

Write-Host "Creating environment..."
conda env create -n $EnvName -f $EnvFile

Write-Host "Installing CUDA-enabled PyTorch wheels into $EnvName"
Write-Host "First removing any existing CPU-only torch packages from the env"
& conda run -n $EnvName python -m pip uninstall -y torch torchvision torchaudio

Write-Host "Installing PyTorch nightly cu128 wheel (explicit CUDA index)"
& conda run -n $EnvName python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
if ($LASTEXITCODE -ne 0) {
	Write-Host "Nightly cu128 install failed. Re-try with the latest stable CUDA wheel if available or share the pip output."
	exit $LASTEXITCODE
}

Write-Host "Optional: reinstall torchvision/torchaudio later if you actually need them. The main training code only requires torch."

Write-Host "Verifying PyTorch GPU / arch support"
@'
import sys
try:
	import torch
except Exception as e:
	print('torch import failed:', e)
	sys.exit(2)
print('torch.__version__=', torch.__version__)
print('torch.version.cuda=', getattr(torch.version, 'cuda', None))
print('cuda available=', torch.cuda.is_available())
if not torch.cuda.is_available():
	print('CUDA is not available; this is not a usable GPU build for Blackwell.')
	sys.exit(3)
try:
	print('device capability=', torch.cuda.get_device_capability(0))
except Exception as e:
	print('get_device_capability error:', e)
try:
	archs = getattr(torch.cuda, 'get_arch_list', None)
	print('arch list=', archs() if archs else 'N/A')
except Exception as e:
	print('get_arch_list error:', e)
'@ | conda run -n $EnvName python -

Write-Host "If the arch list includes 'sm_120' and cuda available is True, you have a working Blackwell-capable env."

