import argparse
import os
import ssl

import uvicorn

from acps_sdk.aip.aip_peer_cert import AipPeerCertH11Protocol

ROOT = os.path.dirname(os.path.abspath(__file__))
PRIVATE = os.path.join(ROOT, "private")
TRUST = os.path.join(PRIVATE, "trust-bundle.pem")

AGENTS = {
    "preprocess": {
        "module": "preprocess_partner:app",
        "port": 8013,
        "aic": "1.2.156.3088.1.BUPT.29ZHHL.QMEFXN.PU0PW3.0YUA",
        "cert": "preprocess-server-cert.pem",
        "key": "preprocess-server-key.pem",
    },
    "detect": {
        "module": "chip_partner:app",
        "port": 8012,
        "aic": "1.2.156.3088.1.BUPT.29ZHHL.5S9JW0.5E1WJU.0E4Y",
        "cert": "detect-server-cert.pem",
        "key": "detect-server-key.pem",
    },
    "report": {
        "module": "report_partner:app",
        "port": 8015,
        "aic": "1.2.156.3088.1.BUPT.29ZHHL.N0DUXP.IYHWFE.0N9N",
        "cert": "report-server-cert.pem",
        "key": "report-server-key.pem",
    },
    "leader": {
        "module": "chipinspect.server:app",
        "port": 8017,
        "aic": "1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4",
        "cert": "leader-server-cert.pem",
        "key": "leader-server-key.pem",
    },
    "feedback": {
        "module": "feedback_partner:app",
        "port": 8016,
        "aic": "1.2.156.3088.1.BUPT.29ZHHL.LU5GGY.8LFBAM.0SMV",
        "cert": "feedback-server-cert.pem",
        "key": "feedback-server-key.pem",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("agent", choices=sorted(AGENTS.keys()))
args = parser.parse_args()
cfg = AGENTS[args.agent]

os.chdir(ROOT)
os.environ["CHIP_AGENT_AIC"] = cfg["aic"]
os.environ["CHIP_IDENTITY_BINDING"] = "1"

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(os.path.join(PRIVATE, cfg["cert"]), os.path.join(PRIVATE, cfg["key"]))
ctx.load_verify_locations(TRUST)
ctx.verify_mode = ssl.CERT_REQUIRED
ctx.minimum_version = ssl.TLSVersion.TLSv1_2

print("starting", args.agent, "on", cfg["port"], "aic", cfg["aic"], flush=True)

config = uvicorn.Config(
    app=cfg["module"],
    host="127.0.0.1",
    port=cfg["port"],
    http=AipPeerCertH11Protocol,
    log_level="info",
    proxy_headers=False,
)
config.load()
config.ssl = ctx
uvicorn.Server(config).run()
