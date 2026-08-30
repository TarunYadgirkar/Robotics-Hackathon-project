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
        self.scan_summary: Optional[Dict[str, Any]] = None
        self.scan_points = None          # numpy array, downsampled for the viewer
        self.registration: Optional[Dict[str, Any]] = None
        self.kinematics = None
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

    def merge_scan(self, fresh, margin: float = 0.05):
        """Fold a new sweep into the stored one, replacing what it re-observed.

        Re-scanning a patch has to *override* it, not add to it: the usual reason
        to re-scan is that something transient was captured -- a hand, an arm, a
        person -- and simply unioning the clouds keeps the ghost forever. Points
        inside the new sweep's bounding box are dropped before the new ones go
        in, so whatever was there before is replaced by what is there now.

        A full-room re-scan therefore replaces everything, which is what one
        would expect, and a small patch touches only that patch.

        This does NOT remove something that floated in front of a surface -- a
        hand held between the phone and a wall leaves points at the hand's depth,
        and re-scanning the wall produces points at the wall's depth, whose
        bounding box need never contain the hand. Removing that needs either
        free-space carving along the camera rays or the explicit erase below.
        """
        import numpy as np

        fresh = np.asarray(fresh, dtype=float).reshape(-1, 3)
        if self.scan_points is None or len(self.scan_points) == 0 or len(fresh) == 0:
            return fresh, 0

        existing = np.asarray(self.scan_points, dtype=float)
        low = fresh.min(axis=0) - margin
        high = fresh.max(axis=0) + margin
        inside = np.all((existing >= low) & (existing <= high), axis=1)

        kept = existing[~inside]
        return np.vstack([kept, fresh]), int(inside.sum())

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
                if route == "/api/scan_summary":
                    self._send_json(server.scan_summary or {})
                    return
                if route == "/api/scan_points":
                    if server.scan_points is None:
                        self._send_json({"points": [], "registered": False})
                        return
                    points = server.scan_points
                    if server.registration is not None:
                        import numpy as np

                        rotation = np.array(server.registration["rotation"])
                        translation = np.array(server.registration["translation"])
                        points = points @ rotation.T + translation
                    self._send_json({
                        "points": [round(float(v), 4) for v in points.ravel()],
                        "registered": server.registration is not None,
                        "rmse_mm": None if server.registration is None else server.registration["rmse_mm"],
                    })
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
                if self.path.startswith("/api/align_seed"):
                    self._align_seed()
                    return
                if self.path.startswith("/api/scan_erase"):
                    self._erase()
                    return
                if self.path.startswith("/api/register"):
                    self._register()
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

            def _align_seed(self):
                """Align the scan using the arm's known shape, seeded by one tap.

                The tap only has to land within a hand's width of the arm; the
                precision comes from ICP over thousands of known surface points
                afterwards. A global search without a seed is what does not work:
                the arm is a small object in a room-sized cloud, and
                point-to-cloud distance alone will park it against any wall.
                """
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    seed = payload["seed"]
                    pose = payload["pose"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    self._send_json({"error": "expected {seed: [x,y,z], pose: [6 joint angles]}"}, 400)
                    return

                if server.scan_points is None or not len(server.scan_points):
                    self._send_json({"error": "no scan uploaded yet"}, 400)
                    return

                from yam.kinematics import YamKinematics
                from yam.scan_registration import dense_arm_surface, refine_from_seed

                kinematics = server.kinematics or YamKinematics()
                model = dense_arm_surface(kinematics, pose, max_points=3000)
                result = refine_from_seed(server.scan_points, model, seed)

                if result is None:
                    self._send_json({"error": "no fit found near that point"}, 400)
                    return

                server.registration = {
                    "rotation": result.rotation.tolist(),
                    "translation": result.translation.tolist(),
                    "rmse_mm": round(result.rmse * 1000, 1),
                    "pairs": 0,
                    "inliers": result.inliers,
                    "model_points": result.model_points,
                    "trustworthy": bool(result.is_trustworthy),
                    "method": "arm-shape ICP from a seed",
                }
                self._send_json(server.registration)

            def _erase(self):
                """Delete scanned geometry inside a sphere.

                The direct answer to a transient object caught in the sweep: the
                operator can see the artefact, so let them point at it, rather
                than inferring it from geometry that cannot distinguish a hand
                from a shelf.
                """
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    centre = payload["centre"]
                    radius = float(payload.get("radius", 0.25))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self._send_json({"error": "expected {centre: [x,y,z], radius: r}"}, 400)
                    return

                if server.scan_points is None or not len(server.scan_points):
                    self._send_json({"error": "no scan to erase from"}, 400)
                    return

                import numpy as np

                points = np.asarray(server.scan_points, dtype=float)
                keep = np.linalg.norm(points - np.array(centre, dtype=float), axis=1) > radius
                removed = int((~keep).sum())
                server.scan_points = points[keep]
                self._send_json({"removed": removed, "remaining": int(keep.sum())})

            def _register(self):
                """Align the uploaded scan to the robot frame from paired points."""
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send_json({"error": "bad json"}, 400)
                    return

                try:
                    import numpy as np

                    from yam.lidar import kabsch

                    result = kabsch(np.array(payload["scan"]), np.array(payload["robot"]))
                except Exception as error:
                    self._send_json({"error": str(error)}, 400)
                    return

                server.registration = {
                    "rotation": result.rotation.tolist(),
                    "translation": result.translation.tolist(),
                    "rmse_mm": round(result.rmse * 1000, 1),
                    "pairs": len(payload["scan"]),
                    "trustworthy": bool(result.is_trustworthy),
                }
                self._send_json(server.registration)

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

                summary = {"saved": destination, "bytes": length - remaining}
                try:
                    import numpy as np

                    from yam.lidar import load_point_cloud

                    points = load_point_cloud(destination)
                    summary["points"] = int(len(points))
                    summary["extent_m"] = [round(float(v), 3) for v in (points.max(axis=0) - points.min(axis=0))]
                    # Thin it for the browser: a phone scan is hundreds of
                    # thousands of points and the viewer only needs enough to
                    # recognise the room and click a feature.
                    step = max(1, len(points) // 40000)
                    fresh = points[::step]

                    merged, replaced = server.merge_scan(fresh)
                    summary["points_kept"] = int(len(merged))
                    summary["points_replaced"] = int(replaced)
                    server.scan_points = merged
                    server.registration = None
                    server.scan_summary = summary
                except Exception as error:
                    # A scan we cannot parse is worth reporting, not worth failing
                    # the upload over -- the file is on disk either way.
                    summary["parse_error"] = str(error)
                    server.scan_summary = summary
                self._send_json(summary)

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
