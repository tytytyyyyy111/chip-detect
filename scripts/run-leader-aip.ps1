$agent = $PSScriptRoot



$root = Split-Path -Parent $agent



$env:YOLO_CONFIG_DIR = $agent



$env:YOLO_OFFLINE = "true"



$env:PYTHONPATH = Join-Path $root "chip-inspection\src"



$env:CHIP_MODEL_PATH = Join-Path $root "chip-inspection\runs\detect\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"



$env:CHIP_MTLS_ENABLED = "1"



$env:CHIP_LEADER_AIC = "1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4"



$env:CHIP_LEADER_CERT = Join-Path $agent "private\leader-cert.pem"



$env:CHIP_LEADER_KEY = [IO.Path]::Combine($agent, "private", "leader-key.pem")



$env:CHIP_TRUST_BUNDLE = Join-Path $agent "private\trust-bundle.pem"



$env:CHIP_AIC_PREPROCESS = "1.2.156.3088.1.BUPT.29ZHHL.QMEFXN.PU0PW3.0YUA"



$env:CHIP_AIC_DETECT = "1.2.156.3088.1.BUPT.29ZHHL.5S9JW0.5E1WJU.0E4Y"



$env:CHIP_AIC_REPORT = "1.2.156.3088.1.BUPT.29ZHHL.N0DUXP.IYHWFE.0N9N"



$env:CHIP_AIC_FEEDBACK = "1.2.156.3088.1.BUPT.29ZHHL.LU5GGY.8LFBAM.0SMV"



Set-Location $agent



.\.venv\Scripts\python.exe secure_server.py leader