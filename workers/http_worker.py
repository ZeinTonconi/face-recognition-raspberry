"""
workers/http_worker.py — Sends attendance POST requests in a background thread.

Non-blocking: the main loop never waits for network responses.
"""

import queue
import threading
from datetime import datetime, timezone

import requests

from config import SERVER_URL, DEVICE_ID, HTTP_TIMEOUT


class HttpWorker(threading.Thread):
    """
    Background thread that POSTs attendance records to the server.

    Callbacks:
        on_success(name)         called on HTTP 200 / 201 / 409
        on_failure(name, reason) called on connection error / timeout / 4xx-5xx
    """

    def __init__(self, on_success, on_failure):
        super().__init__(daemon=True)
        self._q         = queue.Queue()
        self._running   = True
        self.on_success = on_success
        self.on_failure = on_failure

    def send(self, name: str, confidence: float) -> None:
        """Queue an attendance record for sending."""
        self._q.put({
            "nombre":       name,
            "accurrancy": round(confidence, 2),
            "fecha":  datetime.now(timezone.utc).isoformat(),
            "device_id":  DEVICE_ID,
        })

    def run(self) -> None:
        while self._running:
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            name = payload["nombre"]
            print(payload)
            try:
                resp = requests.post(SERVER_URL, json=payload,
                                     timeout=HTTP_TIMEOUT)
                if resp.status_code in (200, 201, 409):
                    self.on_success(name)
                else:
                    self.on_failure(name, f"HTTP {resp.status_code}")
            except requests.exceptions.ConnectionError:
                self.on_failure(name, "Server unreachable")
            except requests.exceptions.Timeout:
                self.on_failure(name, "Request timed out")
            except Exception as exc:
                self.on_failure(name, str(exc))

    def stop(self) -> None:
        self._running = False
