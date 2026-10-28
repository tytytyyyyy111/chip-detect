$ErrorActionPreference = "Continue"

$agent = $PSScriptRoot
$root = Split-Path -Parent $agent
$python = Join-Path $agent ".venv\Scripts\python.exe"

function Stop-Port([int]$port) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
}

# 环境变量必须在启动子进程之前设置，否则子进程继承不到
$tokenFile = Join-Path $root "frp\config\url_token.txt"
if (Test-Path $tokenFile) { $env:CHIP_URL_TOKEN = (Get-Content $tokenFile -Raw).Trim() }

$env:CHIP_MTLS_ENABLED = "1"
$env:CHIP_LEADER_AIC = "1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4"
$env:CHIP_LEADER_CERT = Join-Path $agent "private\leader-cert.pem"
$env:CHIP_LEADER_KEY = Join-Path $agent "private\leader-key.pem"
$env:CHIP_TRUST_BUNDLE = Join-Path $agent "private\trust-bundle.pem"
$env:CHIP_AIC_PREPROCESS = "1.2.156.3088.1.BUPT.29ZHHL.QMEFXN.PU0PW3.0YUA"
$env:CHIP_AIC_DETECT = "1.2.156.3088.1.BUPT.29ZHHL.5S9JW0.5E1WJU.0E4Y"
$env:CHIP_AIC_REPORT = "1.2.156.3088.1.BUPT.29ZHHL.N0DUXP.IYHWFE.0N9N"
$env:CHIP_AIC_FEEDBACK = "1.2.156.3088.1.BUPT.29ZHHL.LU5GGY.8LFBAM.0SMV"

$env:PYTHONPATH = Join-Path $root "chip-inspection\src"
$env:CHIP_MODEL_PATH = Join-Path $root "chip-inspection\runs\detect\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"
$env:YOLO_CONFIG_DIR = $agent
$env:YOLO_OFFLINE = "true"

foreach ($p in 8012,8013,8014,8015,8016,8017) { Stop-Port $p }
Start-Sleep -Seconds 2

Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","detect"
Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","preprocess"
Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","report"
Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","feedback"
Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","leader"
Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chipinspect.server:app","--host","127.0.0.1","--port","8014"

Start-Sleep -Seconds 10
foreach ($p in 8012,8013,8014,8015,8016,8017) {
    $ok = [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
    Write-Output ("port " + $p + " -> " + $ok)
}
