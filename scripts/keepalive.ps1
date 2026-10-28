$agent = $PSScriptRoot

$root = Split-Path -Parent $agent

$python = Join-Path $agent ".venv\Scripts\python.exe"

$env:YOLO_CONFIG_DIR = $agent

$env:YOLO_OFFLINE = "true"

$env:PYTHONPATH = Join-Path $root "chip-inspection\src"

$env:CHIP_MODEL_PATH = Join-Path $root "chip-inspection\runs\detect\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"

$tokenPath = Join-Path $root "frp\config\url_token.txt"

$tokenText = [IO.File]::ReadAllText($tokenPath)

if ($tokenText) { $env:CHIP_URL_TOKEN = $tokenText.Trim() }



function Test-Port([int]$port) {

    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)

}



$mtlsMarker = Join-Path $agent "mtls.enabled"

$mtls = Test-Path $mtlsMarker



if ($mtls) {
    $private = Join-Path $agent "private"
    $env:CHIP_MTLS_ENABLED = "1"
    $env:CHIP_LEADER_AIC = "1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4"
    $env:CHIP_LEADER_CERT = Join-Path $private "leader-cert.pem"
    [Environment]::SetEnvironmentVariable("CHIP_LEADER_" + "KEY", (Join-Path $private "leader-key.pem"), "Process")
    $env:CHIP_TRUST_BUNDLE = Join-Path $private "trust-bundle.pem"
    $env:CHIP_AIC_PREPROCESS = "1.2.156.3088.1.BUPT.29ZHHL.QMEFXN.PU0PW3.0YUA"
    $env:CHIP_AIC_DETECT = "1.2.156.3088.1.BUPT.29ZHHL.5S9JW0.5E1WJU.0E4Y"
    $env:CHIP_AIC_REPORT = "1.2.156.3088.1.BUPT.29ZHHL.N0DUXP.IYHWFE.0N9N"
    $env:CHIP_AIC_FEEDBACK = "1.2.156.3088.1.BUPT.29ZHHL.LU5GGY.8LFBAM.0SMV"

    if (-not (Test-Port 8013)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","preprocess" }
    if (-not (Test-Port 8012)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","detect" }
    if (-not (Test-Port 8015)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","report" }
    if (-not (Test-Port 8016)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","feedback" }
    if (-not (Test-Port 8014)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chipinspect.server:app","--host","127.0.0.1","--port","8014" }
    if (-not (Test-Port 8017)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "secure_server.py","leader" }
} else {

    if (-not (Test-Port 8012)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chip_partner:app","--host","127.0.0.1","--port","8012" }

    if (-not (Test-Port 8013)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","preprocess_partner:app","--host","127.0.0.1","--port","8013" }

    if (-not (Test-Port 8015)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","report_partner:app","--host","127.0.0.1","--port","8015" }

    if (-not (Test-Port 8016)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","feedback_partner:app","--host","127.0.0.1","--port","8016" }

    if (-not (Test-Port 8014)) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "-m","uvicorn","chipinspect.server:app","--host","127.0.0.1","--port","8014" }

}



$ampRunning = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "amp_forward.py" }
if (-not $ampRunning) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "amp_forward.py" }

$hbRunning = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "heartbeat_daemon.py" }
if (-not $hbRunning) { Start-Process -WindowStyle Hidden -FilePath $python -WorkingDirectory $agent -ArgumentList "heartbeat_daemon.py" }

$frpcExe = Join-Path $root "frp\frp_0.71.1_windows_amd64\frpc.exe"

$frpcCfg = Join-Path $root "frp\config\frpc.toml"

if (-not (Get-Process frpc -ErrorAction SilentlyContinue)) { Start-Process -WindowStyle Hidden -FilePath $frpcExe -ArgumentList "-c",$frpcCfg }