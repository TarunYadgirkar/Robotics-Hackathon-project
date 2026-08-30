"""Tiny HTTP server backing the live enrollment viewer.

Deliberately stdlib-only: this runs next to a robot at a hackathon, and a viewer
that fails because a web framework did not install is a viewer that does not
exist. The enrollment script pushes state in; the browser polls it and posts
back button presses, so the operator drives from the same screen they are
watching, instead of looking away to a terminal.
"""

import json
import os
import queue
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

# web/enroll/, not web/: web/ is the team's Vite app. A subdirectory is outside
# its build graph (only web/index.html is an entry), so the two coexist.
WEB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "enroll")


class VizServer:
    def __init__(self, port: int = 8420, web_root: str = WEB_ROOT):
        self.port = port
        self.web_root = web_root
        self.static_payloads: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {"status": "starting"}
        self._lock = threading.Lock()
        self.commands: "queue.Queue[str]" = queue.Queue()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def update(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state = state

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self) -> str:
        server = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=server.web_root, **kwargs)

            def log_message(self, *args):
                pass  # the terminal belongs to the enrollment prompts

            def _send_json(self, payload: Dict[str, Any], code: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                route = self.path.split("?")[0]
                if route == "/api/state":
                    self._send_json(server.snapshot())
                    return
                if route.startswith("/api/static/"):
                    key = route[len("/api/static/"):]
                    if key in server.static_payloads:
                        self._send_json(server.static_payloads[key])
                    else:
                        self._send_json({"error": f"no payload {key!r}"}, 404)
                    return
                if self.path == "/":
                    self.path = "/enroll.html"
                super().do_GET()

            def do_POST(self):
                if self.path != "/api/command":
                    self._send_json({"error": "unknown endpoint"}, 404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send_json({"error": "bad json"}, 400)
                    return
                action = str(payload.get("action", ""))
                if action:
                    server.commands.put(action)
                self._send_json({"accepted": bool(action)})

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self.port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def next_command(self, timeout: float = 0.0) -> Optional[str]:
        try:
            return self.commands.get(timeout=timeout) if timeout else self.commands.get_nowait()
        except queue.Empty:
            return None
