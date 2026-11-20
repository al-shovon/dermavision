# One-time setup for Windows. Run it from this folder in PowerShell:  .\setup.ps1
# If PowerShell blocks the script:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

Write-Host "== Backend: virtual environment and packages =="
Set-Location backend
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
Check "Creating the virtual environment"
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
Check "Upgrading pip"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Check "Installing PyTorch"
& $py -m pip install -r requirements.txt
Check "Installing the backend requirements"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Set-Location ..

Write-Host "== Frontend: npm packages =="
Set-Location frontend
npm install
Check "npm install"
if (-not (Test-Path .env.local)) { Copy-Item .env.local.example .env.local }
Set-Location ..

Write-Host ""
Write-Host "Setup finished."
Write-Host "In VS Code: select the Python interpreter (backend\.venv), then press Ctrl+Shift+B to start the API and the site."
