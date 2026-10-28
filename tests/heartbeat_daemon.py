"""AMP 心跳守护进程：为 5 个智能体周期上报 heartbeat。

平台按 AMP 心跳判定智能体"在线"，心跳缺失会导致发现服务检索不到。
本进程每 15 秒为每个 AIC 追加一条 amp_heartbeat_<name>.jsonl，
由 amp_forward.py 转发到 Kafka 主题 amp.heartbeat。
"""

import asyncio
import os
import socket
import sys
from pathlib import Path

from acps_sdk.amp.heartbeat_emitter import HeartbeatEmitter

LOG_DIR = Path(os.environ.get("AMP_LOG_DIR", str(Path(__file__).resolve().parent / "logs")))
INTERVAL = float(os.environ.get("AMP_HEARTBEAT_INTERVAL", "15"))
LOCK_PORT = int(os.environ.get("AMP_HEARTBEAT_LOCK_PORT", "19998"))

AGENTS = {
    "leader": "1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4",
    "preprocess": "1.2.156.3088.1.BUPT.29ZHHL.QMEFXN.PU0PW3.0YUA",
    "detect": "1.2.156.3088.1.BUPT.29ZHHL.5S9JW0.5E1WJU.0E4Y",
    "report": "1.2.156.3088.1.BUPT.29ZHHL.N0DUXP.IYHWFE.0N9N",
    "feedback": "1.2.156.3088.1.BUPT.29ZHHL.LU5GGY.8LFBAM.0SMV",
}


def acquire_singleton():
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", LOCK_PORT))
        sock.listen(1)
    except OSError:
        print("another heartbeat daemon is running", flush=True)
        raise SystemExit(0)
    return sock


async def main():
    _lock = acquire_singleton()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name, aic in AGENTS.items():
        emitter = HeartbeatEmitter(LOG_DIR / f"amp_heartbeat_{name}.jsonl", aic)
        tasks.append(asyncio.create_task(emitter.run_periodic(INTERVAL)))
        print(f"heartbeat scheduled: {name} {aic[-12:]} every {INTERVAL}s", flush=True)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped", flush=True)
