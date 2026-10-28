import json, os, socket, time
from pathlib import Path
from kafka import KafkaProducer

SINGLETON = socket.socket()
try:
    SINGLETON.bind(("127.0.0.1", 19999))
    SINGLETON.listen(1)
except OSError:
    print("another forwarder is running", flush=True)
    raise SystemExit(0)

LOG_DIR = Path(os.environ.get("AMP_LOG_DIR", r"%CHIP_ROOT%\chip-agent\logs"))
STATE = LOG_DIR / "amp_forward_state.json"
TOPIC = os.environ.get("AMP_ACCESS_TOPIC", "amp.access")
MESSAGE_TOPIC = os.environ.get("AMP_MESSAGE_TOPIC", "amp.message")
HEARTBEAT_TOPIC = os.environ.get("AMP_HEARTBEAT_TOPIC", "amp.heartbeat")
BOOTSTRAP = os.environ.get("AMP_KAFKA_BOOTSTRAP", "wt.ioa.pub:19092")

def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state):
    STATE.write_text(json.dumps(state), encoding="utf-8")

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: v,
        acks=1,
        linger_ms=200,
        request_timeout_ms=15000,
    )
    state = load_state()
    print("forwarder started", BOOTSTRAP, TOPIC, flush=True)
    while True:
        files = sorted(LOG_DIR.glob("amp_*.jsonl"))
        for f in files:
            if f.name.startswith("amp_access_"):
                topic = TOPIC
            elif f.name.startswith("amp_message_"):
                topic = MESSAGE_TOPIC
            elif f.name.startswith("amp_heartbeat_"):
                topic = HEARTBEAT_TOPIC
            else:
                continue
            key = f.name
            offset = int(state.get(key, 0))
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    while True:
                        line = fh.readline()
                        if not line:
                            break
                        if line.strip():
                            producer.send(topic, line.strip().encode("utf-8"))
                        offset = fh.tell()
                state[key] = offset
                save_state(state)
            except Exception as e:
                print("read error", key, type(e).__name__, str(e)[:120], flush=True)
        producer.flush(timeout=5)
        time.sleep(2)

if __name__ == "__main__":
    main()
