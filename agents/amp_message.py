import os
from pathlib import Path

from acps_sdk.amp import MessageEmitter
from acps_sdk.amp.models import MessageBody, MessageDestination

LOG_DIR = Path(os.getenv("AMP_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_EMITTERS = {}


def make_emitter(name: str, aic: str) -> MessageEmitter:
    if name not in _EMITTERS:
        _EMITTERS[name] = MessageEmitter(LOG_DIR / ("amp_message_%s.jsonl" % name), aic=aic)
    return _EMITTERS[name]


def message_body(event_type: str, message_id: str, destination: str, payload_size: int = 0) -> MessageBody:
    return MessageBody(
        event_type=event_type,
        system="aip-rpc",
        destination=MessageDestination(name=destination, kind="queue"),
        message_id=message_id,
        payload_size_bytes=payload_size,
    )
