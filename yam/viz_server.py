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
import re
import secrets
import socket
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

# web/enroll/, not web/: web/ is the team's Vite app. A subdirectory is outside
# its build graph (only web/index.html is an entry), so the two coexist.
WEB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "enroll")


class VizServer:
    def __init__(self, port: int = 8420, web_root: str = WEB_ROOT, host: str = "0.0.0.0",
                 upload_dir: Optional[str] = None):
        self.port = port
        self.host = host
        self.web_root = web_root
        self.upload_dir = upload_dir or os.getcwd()
        self.static_payloads: Dict[str, Any] = {}
        self.uploads: list = []
        #: Guards the API once the server is reachable beyond this machine. The
        #: page is served freely so it can load and read the token out of its own
        #: URL; the endpoints that capture points and write files are not.
        self.token = secrets.token_urlsafe(12)
        self._tunnel: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
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

    @staticmethod
    def lan_address() -> Optional[str]:
        """This machine's address on the local network.

        Opening a UDP socket toward a public address makes the OS choose the
        outbound interface; no packet is actually sent. `gethostname()` is
        unreliable here -- on macOS it often resolves to a .local name the phone
        cannot look up.
        """
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            return None
        finally:
            probe.close()

    def urls(self) -> Dict[str, Optional[str]]:
        address = self.lan_address()
        return {
            "local": f"http://127.0.0.1:{self.port}/",
            "lan": f"http://{address}:{self.port}/" if address and self.host != "127.0.0.1" else None,
        }

    def start(self) -> str:
        server = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=server.web_root, **kwargs)

            def log_message(self, *args):
                pass  # the terminal belongs to the enrollment prompts

            def _authorized(self) -> bool:
                if self.client_address[0] in ("127.0.0.1", "::1"):
                    return True
                supplied = self.headers.get("X-Token")
                if not supplied and "?" in self.path:
                    from urllib.parse import parse_qs, urlparse

                    supplied = (parse_qs(urlparse(self.path).query).get("k") or [None])[0]
                return secrets.compare_digest(str(supplied or ""), server.token)

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
                if route.startswith("/api/") and not self._authorized():
                    self._send_json({"error": "unauthorized"}, 401)
                    return
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
                if not self._authorized():
                    self._send_json({"error": "unauthorized"}, 401)
                    return
                if self.path.startswith("/api/scan"):
                    self._receive_scan()
                    return
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

            def _receive_scan(self):
                """Accept a scan file uploaded from the phone that captured it."""
                name = os.path.basename(self.headers.get("X-Filename", "scan.ply")) or "scan.ply"
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    self._send_json({"error": "empty upload"}, 400)
                    return

                destination = os.path.join(server.upload_dir, name)
                remaining = length
                with open(destination, "wb") as handle:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)

                server.uploads.append(destination)
                server.commands.put("scan_uploaded")
                self._send_json({"saved": destination, "bytes": length - remaining})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self.port}/"

    def start_tunnel(self, timeout: float = 25.0) -> Optional[str]:
        """Expose the viewer through a Cloudflare quick tunnel.

        Needed on networks with client isolation -- eduroam and most guest wifi
        will not route phone-to-laptop traffic at all, so a LAN address cannot
        work no matter how the server is bound. The tunnel URL is public, which
        is why the API is token-guarded.
        """
        try:
            self._tunnel = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{self.port}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except FileNotFoundError:
            return None

        pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        deadline = time.time() + timeout
        while time.time() < deadline and self._tunnel.poll() is None:
            line = self._tunnel.stderr.readline()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                self.tunnel_url = match.group(0)
                # Keep draining stderr, or cloudflared blocks on a full pipe.
                threading.Thread(target=lambda: [None for _ in self._tunnel.stderr], daemon=True).start()
                return self.tunnel_url
        return None

    def stop(self) -> None:
        if self._tunnel is not None:
            self._tunnel.terminate()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def next_command(self, timeout: float = 0.0) -> Optional[str]:
        try:
            return self.commands.get(timeout=timeout) if timeout else self.commands.get_nowait()
        except queue.Empty:
            return None
