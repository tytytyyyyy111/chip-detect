$ErrorActionPreference = "Continue"

$agent = $PSScriptRoot
$root = Split-Path -Parent $agent
$python = Join-Path $agent ".venv\Scripts\python.exe"

function Test-Port([int]$port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

$env:YOLO_CONFIG_DIR = $agent
$env:YOLO_OFFLINE = "true"

$tokenFile = Join-Path $root "frp\config\url_token.txt"
if (Test-Path $tokenFile) { $env:CHIP_URL_TOKEN = (Get-Content $tokenFile -Raw).Trim() }

if (-not (Test-Port 8012)) {
    Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chip_partner:app","--host","127.0.0.1","--port","8012"
}
if (-not (Test-Port 8013)) {
    Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","preprocess_partner:app","--host","127.0.0.1","--port","8013"
}
if (-not (Test-Port 8015)) {
    Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","report_partner:app","--host","127.0.0.1","--port","8015"
}
if (-not (Test-Port 8016)) {
    Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","feedback_partner:app","--host","127.0.0.1","--port","8016"
}
if (-not (Test-Port 8014)) {
    $env:PYTHONPATH = Join-Path $root "chip-inspection\src"
    $env:CHIP_MODEL_PATH = Join-Path $root "chip-inspection\runs\detect\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"
    Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chipinspect.server:app","--host","127.0.0.1","--port","8014"
}

$frpc = Join-Path $root "frp\frp_0.71.1_windows_amd64\frpc.exe"
$frpcCfg = Join-Path $root "frp\config\frpc.toml"
if (-not (Get-Process frpc -ErrorAction SilentlyContinue)) {
    Start-Process -WindowStyle Hidden -FilePath $frpc -ArgumentList "-c",$frpcCfg
}

Start-Sleep -Seconds 5
foreach ($p in 8012,8013,8014,8015,8016) {
    Write-Output ("port " + $p + " -> " + (Test-Port $p))
}
Write-Output ("frpc -> " + [bool](Get-Process frpc -ErrorAction SilentlyContinue))
