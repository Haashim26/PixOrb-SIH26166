$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# CPU-safe PyTorch installation; CUDA is optional and can be substituted for GPU systems.
& .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install git+https://github.com/cvg/LightGlue.git
Write-Host 'PixOrb environment setup complete.'
