"""Show the raw scan, let the operator click the arm, then register on that seed.

Registration is the only thing standing between a full LiDAR model of the
workcell and using it for planning. Automatic search over a room-sized cloud did
not work -- the arm is a small object and point-to-cloud distance alone will park
it against any wall -- but one click removes the search entirely. Precision then
comes from ICP over the arm's known surface, not from the click.
"""

import argparse
import hashlib
import json
import os
import queue
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from yam.enrollment import EnrollmentSession, recompute_positions
from yam.kinematics import YamKinematics
from yam.lidar import load_point_cloud, scan_timestamp_from_path
from yam.mesh_export import export_arm_meshes
from yam.scan_registration import dense_arm_surface, refine_from_seed

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Where is the arm?</title>
<style>
 html,body{height:100%;margin:0;background:#fff;color:#171214;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif}
 canvas{display:block} #s{position:fixed;inset:0;width:100vw;height:100vh}
 header{position:fixed;top:0;left:0;right:0;padding:20px 24px;pointer-events:none}
 h1{margin:0;font-size:20px;font-weight:600}
 p{margin:6px 0 0;color:#6E6469;font-size:13px}
 #st{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);background:#171214;color:#fff;
   font-size:13px;padding:10px 18px;border-radius:99px}
</style></head><body>
<canvas id="s"></canvas>
<header><h1>Click the arm in the scan</h1>
<p>Drag to orbit, scroll to zoom. Roughly is fine &mdash; a hand&rsquo;s width is close enough.</p></header>
<div id="st">loading scan&hellip;</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const canvas=document.getElementById('s');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0xffffff,1);
const scene=new THREE.Scene(); const camera=new THREE.PerspectiveCamera(45,1,0.01,200);
scene.add(new THREE.HemisphereLight(0xffffff,0xeeeeee,1.1));
let cloud=null, centre=new THREE.Vector3(), radius=2;
fetch('/points').then(r=>r.json()).then(d=>{
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(d.points,3));
  g.computeBoundingSphere(); centre.copy(g.boundingSphere.center); radius=g.boundingSphere.radius;
  cloud=new THREE.Points(g,new THREE.PointsMaterial({size:0.012,color:0x8A8F98}));
  scene.add(cloud); target.copy(centre); dist=radius*1.7;
  document.getElementById('st').textContent=d.points.length/3+' points \\u2014 click the arm';
});
let yaw=0.9,pitch=0.35,dist=4; const target=new THREE.Vector3();
let drag=false,lx=0,ly=0,moved=0;
canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;moved=0});
addEventListener('pointerup',e=>{ if(drag&&moved<6) pick(e.clientX,e.clientY); drag=false; });
addEventListener('pointermove',e=>{if(!drag)return;moved+=Math.abs(e.clientX-lx)+Math.abs(e.clientY-ly);
  yaw-=(e.clientX-lx)*.006;pitch=Math.max(-1.4,Math.min(1.4,pitch-(e.clientY-ly)*.006));lx=e.clientX;ly=e.clientY});
canvas.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(.3,dist*(1+Math.sign(e.deltaY)*.1))},{passive:false});
const ray=new THREE.Raycaster(); ray.params.Points.threshold=0.03;
function pick(x,y){
  if(!cloud) return;
  ray.setFromCamera(new THREE.Vector2((x/innerWidth)*2-1,-(y/innerHeight)*2+1),camera);
  const h=ray.intersectObject(cloud);
  if(!h.length){document.getElementById('st').textContent='no point there \\u2014 try again';return;}
  const p=h[0].point;
  const m=new THREE.Mesh(new THREE.SphereGeometry(0.05,16,12),new THREE.MeshBasicMaterial({color:0xC2183C}));
  m.position.copy(p); scene.add(m);
  document.getElementById('st').textContent='seed at height '+p.y.toFixed(2)+' m \\u2014 registering\\u2026';
  fetch('/seed',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({seed:[p.x,p.y,p.z]})}).then(()=>poll());
}
let fitted=null;
function poll(){
  fetch('/fit').then(r=>r.json()).then(d=>{
    if(!d.ready){ setTimeout(poll,500); return; }
    if(fitted) scene.remove(fitted);
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(d.model,3));
    fitted=new THREE.Points(g,new THREE.PointsMaterial({size:0.02,color:0xC2183C}));
    scene.add(fitted);
    document.getElementById('st').textContent =
      'fit '+d.rmse.toFixed(1)+' mm \u2014 red is where the arm was placed. On the arm? If not, click again.';
  });
}
function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix()}
addEventListener('resize',resize); resize();
(function tick(){const cp=Math.cos(pitch),sp=Math.sin(pitch);
  camera.position.set(target.x+dist*cp*Math.sin(yaw),target.y+dist*sp,target.z+dist*cp*Math.cos(yaw));
  camera.lookAt(target); renderer.render(scene,camera); requestAnimationFrame(tick);})();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--enrollment", default="enrollment.json")
    parser.add_argument("--pose-index", type=int, default=None,
                        help="logged arm pose to use; by default select the pose held when the "
                             "timestamped scan file was uploaded")
    parser.add_argument("--seed", type=float, nargs=3, default=None,
                        help="ARKit x y z seed for a non-interactive one-shot registration")
    parser.add_argument("--port", type=int, default=8460)
    parser.add_argument("--output", default="registration.json")
    args = parser.parse_args()

    kinematics = YamKinematics()
    session = EnrollmentSession.load(args.enrollment)
    recompute_positions(session, kinematics)
    if args.pose_index is not None:
        pose_index = args.pose_index
        try:
            pose_sample = session.pose_log[pose_index]
        except IndexError as error:
            raise SystemExit(f"pose index {pose_index} is outside the logged pose range") from error
    else:
        scan_timestamp = scan_timestamp_from_path(args.scan)
        if scan_timestamp is None:
            raise SystemExit(
                "the scan filename has no timestamp; pass --pose-index for the pose held during the scan"
            )
        try:
            pose_sample, pose_index = session.pose_at(scan_timestamp)
        except ValueError as error:
            raise SystemExit(str(error)) from error

    pose = pose_sample.joint_angles
    measured_surfaces = session.captured_positions()
    scan = load_point_cloud(args.scan)
    print(f"  scan {len(scan):,} points | pose {pose_index} held from "
          f"{pose_sample.timestamp:.3f} | {len(measured_surfaces)} touched surfaces")

    step = max(1, len(scan) // 60000)
    shown = scan[::step]
    seeds: "queue.Queue[list]" = queue.Queue()
    fit_state = {"ready": False}

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/fit":
                body = json.dumps(fit_state).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                self.wfile.write(body); return
            if self.path == "/points":
                body = json.dumps({"points": np.round(shown, 4).ravel().tolist()}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body); return
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seeds.put(json.loads(self.rfile.read(length))["seed"])
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
            self.wfile.write(b"ok")

    if args.seed is None:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{args.port}/"
        print(f"  open {url} and click the arm")
        webbrowser.open(url)
    else:
        seeds.put(args.seed)

    model = dense_arm_surface(kinematics, pose, max_points=3000)
    while True:
        seed = seeds.get()
        fit_state.clear(); fit_state["ready"] = False
        print(f"\n  seed at {np.round(seed, 3).tolist()} -- registering...")
        result = refine_from_seed(scan, model, seed, surface_points=measured_surfaces)
        if result is None:
            if args.seed is not None:
                raise SystemExit("  no fit near the supplied seed")
            print("  no fit near there. Try clicking closer to the middle of the arm.")
            continue
        print(f"  {result.describe()}  [{result.verdict}]")

        if result.is_trustworthy:
            digest = hashlib.sha256()
            with open(args.scan, "rb") as scan_file:
                for chunk in iter(lambda: scan_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            record = {
                "schema_version": 2,
                "rotation": result.rotation.tolist(),
                "translation": result.translation.tolist(),
                "rmse_mm": result.rmse * 1000,
                "model_p95_mm": result.model_p95_error * 1000,
                "inliers": result.inliers,
                "model_points": result.model_points,
                "surface_rmse_mm": result.surface_rmse * 1000,
                "surface_max_error_mm": result.surface_max_error * 1000,
                "surface_points": result.surface_points,
                "surface_spread_m": result.surface_spread,
                "uncertainty_mm": result.uncertainty * 1000,
                "verdict": result.verdict,
                "trustworthy": True,
                "gravity_constrained": True,
                "method": "arm shape plus touched surfaces",
                "scan_filename": os.path.basename(args.scan),
                "scan_sha256": digest.hexdigest(),
                "scan_pose_index": pose_index,
                "scan_pose_timestamp": pose_sample.timestamp,
                "scan_pose": pose,
            }
            with open(args.output, "w") as handle:
                json.dump(record, handle, indent=2)
            print(f"  wrote validated transform to {args.output}")
        else:
            print("  NOT SAVED: the arm and touched surfaces do not jointly validate this transform")

        # Put the fitted model back into scan coordinates so the operator can see
        # exactly where it landed. The stored transform maps scan -> robot as
        # robot = scan @ R.T + t, so the inverse is scan = (robot - t) @ R.
        placed = (model - result.translation) @ result.rotation
        fit_state.clear()
        fit_state.update({
            "ready": True,
            "rmse": float(result.rmse * 1000),
            "model": np.round(placed, 4).ravel().tolist(),
        })
        if result.is_trustworthy and args.seed is None:
            print("  Ctrl-C here, then build the map from the scan.")
        if args.seed is not None:
            if not result.is_trustworthy:
                raise SystemExit(1)
            return


if __name__ == "__main__":
    main()
