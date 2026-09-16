"""Container-only synthetic HTTP protocol exercises for the real analyze process.

No model or outcome service is contacted. Only request hashes and status are retained.
The evaluator copies this driver into the frozen image without modifying its agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCENARIOS = (
    "cold_start",
    "connection_refused",
    "malformed_reply",
    "timeout",
    "budget_exhaustion",
)


def exercise(scenario: str) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("unknown protocol exercise")
    task = json.loads(Path("/input/task.json").read_text())
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024**2:
                self.send_error(400)
                return
            payload = self.rfile.read(size)
            with lock:
                ordinal = len(requests)
                record = {
                    "sequence": ordinal + 1,
                    "request_sha256": hashlib.sha256(payload).hexdigest(),
                    "path": self.path,
                    "action": scenario,
                }
                requests.append(record)
            if scenario == "timeout":
                # The client's read timeout is shorter than this delay.
                time.sleep(1)
                return
            if scenario in ("malformed_reply", "budget_exhaustion"):
                body = b"synthetic invalid JSON"
            else:
                entity = task["entities"][ordinal % len(task["entities"])]["entity_id"]
                text = json.loads(Path(f"/input/corpus/{entity}.json").read_text())[
                    "text"
                ]
                prediction = {
                    "point_forecast": 5.0,
                    "interval": {"level": 0.9, "lo": 4.0, "hi": 6.0},
                    "evidence": [{"doc_id": entity, "quote": text, "claim": text}],
                }
                if task["target"]["type"] == "classification":
                    prediction["label"] = "up"
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": json.dumps(prediction)},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A client timeout has already been recorded as a received request.
                record["client_disconnected"] = True

    server = None
    reserved = None
    if scenario == "connection_refused":
        reserved = socket.socket()
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]  # bound but never listening
    else:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
    # Deliberately avoid inheriting model credentials or developer settings.
    env = {
        key: os.environ[key] for key in ("PATH", "QFBENCH_SEED") if key in os.environ
    }
    env.update(
        {
            "MODEL_ENDPOINT": f"http://127.0.0.1:{port}/v1",
            "MODEL_NAME": "synthetic-protocol",
            "T4_MODEL_TIMEOUT_S": "0.1" if scenario == "timeout" else "2",
            "T4_MODEL_RETRIES": "1" if scenario == "budget_exhaustion" else "2",
            "T4_UNIT_TIMEOUT_S": "60",
            "T4_TOP_K": "3",
            "T4_LOCAL_LLAMA": "0",
        }
    )
    started = time.monotonic()
    try:
        process = subprocess.run(
            [
                sys.executable,
                "/app/analyze.py",
                "analyze",
                "--task",
                "/input/task.json",
                "--corpus",
                "/input/corpus",
                "--out",
                "/output/answer.json",
                "--diagnostics",
                "/output/diagnostics.json",
            ],
            cwd="/app",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )
        return {
            "version": 1,
            "scenario": scenario,
            "returncode": process.returncode,
            "process_elapsed_s": time.monotonic() - started,
            "received_requests": requests,
            "settings": {
                key: value for key, value in env.items() if key.startswith("T4_")
            },
            "scope": "synthetic loopback service; no remote model or production-faithfulness measurement",
        }
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if reserved is not None:
            reserved.close()


if __name__ == "__main__":
    result = exercise(sys.argv[1])
    Path("/output/exercise.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(result["returncode"])
