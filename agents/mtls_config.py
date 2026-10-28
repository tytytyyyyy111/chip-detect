import os

LEADER_AIC = os.getenv("CHIP_LEADER_AIC", "chip-inspection-leader")
LEADER_CERT = os.getenv("CHIP_LEADER_CERT", "")
LEADER_KEY = os.getenv("CHIP_LEADER_KEY", "")
TRUST_BUNDLE = os.getenv("CHIP_TRUST_BUNDLE", "")
URL_TOKEN = os.getenv("CHIP_URL_TOKEN", "")
PARTNER_AICS = {
    "preprocess": os.getenv("CHIP_AIC_PREPROCESS", ""),
    "detect": os.getenv("CHIP_AIC_DETECT", ""),
    "report": os.getenv("CHIP_AIC_REPORT", ""),
    "feedback": os.getenv("CHIP_AIC_FEEDBACK", ""),
}
