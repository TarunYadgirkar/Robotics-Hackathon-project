#!/usr/bin/env python3
"""Local upload API and sequential model pipeline for Factory Atlas."""

from __future__ import annotations

import cgi
import gzip
import hashlib
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker"
PYTHON = WORKER / ".venv" / "bin" / "python"
JOBS_ROOT = ROOT / "work" / "pipeline-jobs"
HAND_MODEL = WORKER / "models" / "hand_landmarker.task"
WILOR_ROOT = ROOT / "work" / "hopformer"
AWS = Path("/Users/carsonsteele/.local/bin/aws")
AWS_PROFILE = os.environ.get("FACTORY3D_AWS_PROFILE", "factory3d-dev")
AWS_REGION = os.environ.get("FACTORY3D_AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("FACTORY3D_S3_BUCKET", "hackathon-video-storage")
CLOUDFRONT = os.environ.get("FACTORY3D_CLOUDFRONT", "https://d1dw8nl6ynliwf.cloudfront.net").rstrip("/")
PORT = int(os.environ.get("FACTORY3D_PIPELINE_PORT", "8788"))

STAGES = [
    ("upload", "Upload source to S3"),
    ("prepare", "Prepare browser video"),
    ("joints", "Hand joints and 3D surface"),
    ("wilor", "Full-length WiLoR mesh"),
    ("overlay", "Render WiLoR video"),
    ("depth", "Full-length depth heat map"),
    ("publish", "Store generated artifacts in S3"),
]

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue: queue.Queue[str] = queue.Queue()


def safe_name(value: str) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "video"
    return stem[:48]


def write_status(job: dict) -> None:
    job["updated_at"] = time.time()
    path = JOBS_ROOT / job["id"] / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2))


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.update(changes)
        write_status(job)


def run(job_id: str, stage: str, command: list[str], cwd: Path = ROOT) -> None:
    update_job(job_id, stage=stage, message=dict(STAGES)[stage])
    log_path = JOBS_ROOT / job_id / "pipeline.log"
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            update_job(job_id, heartbeat=time.time())
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"{dict(STAGES)[stage]} failed (exit {process.returncode})")


def probe_video(path: Path) -> dict:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,pix_fmt,nb_frames:format=duration",
        "-of", "json", str(path),
    ], text=True)
    payload = json.loads(output)
    stream = payload["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    numerator, denominator = (float(part) for part in rate.split("/"))
    fps = numerator / denominator if denominator else 30.0
    duration = float(payload["format"].get("duration") or 0)
    if duration <= 0 or fps <= 0:
        raise RuntimeError("Could not determine video duration and frame rate")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "duration_seconds": duration,
        "codec": stream.get("codec_name"), "pixel_format": stream.get("pix_fmt"),
    }


def aws_cp(job_id: str, source: Path, key: str, content_type: str | None = None) -> None:
    command = [str(AWS), "s3", "cp", str(source), f"s3://{S3_BUCKET}/{key}", "--profile", AWS_PROFILE, "--region", AWS_REGION]
    if content_type:
        command += ["--content-type", content_type]
    command += ["--cache-control", "public,max-age=3600"]
    run(job_id, "upload" if key.startswith("uploads/") else "publish", command)


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    job_dir = JOBS_ROOT / job_id
    source = Path(job["local_source"])
    output = job_dir / "derived"
    output.mkdir(parents=True, exist_ok=True)
    try:
        update_job(job_id, status="running", progress=2, message="Inspecting video")
        metadata = probe_video(source)
        fps = metadata["fps"]
        duration = metadata["duration_seconds"]
        video_id = f"{safe_name(job['filename'])}-{hashlib.sha1((job_id + job['filename']).encode()).hexdigest()[:8]}"
        source_key = f"uploads/{video_id}/source{source.suffix.lower()}"
        derived_key = f"derived/{video_id}"
        source_url = f"{CLOUDFRONT}/{source_key}"
        update_job(job_id, video_id=video_id, metadata=metadata, source_key=source_key, derived_key=derived_key, progress=4)

        aws_cp(job_id, source, source_key, mimetypes.guess_type(source.name)[0] or "video/mp4")
        update_job(job_id, progress=12)

        browser_video = output / "video-browser.mp4"
        run(job_id, "prepare", [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(browser_video),
        ])
        update_job(job_id, progress=20)

        run(job_id, "joints", [
            str(PYTHON), str(WORKER / "process_one_video.py"),
            "--input", str(source), "--output", str(output),
            "--hand-model", str(HAND_MODEL), "--source-url", source_url,
            "--sample-fps", f"{fps:.8f}", "--duration-seconds", f"{duration:.6f}",
        ])
        raw_hands = output / "hands-detections.json"
        (output / "hands.json").replace(raw_hands)
        run(job_id, "joints", [
            str(PYTHON), str(WORKER / "track_and_resample_hands.py"),
            "--input", str(raw_hands), "--output", str(output / "hands.json"),
            "--output-fps", f"{fps:.8f}", "--duration-seconds", f"{duration:.6f}",
        ])
        update_job(job_id, progress=40)

        run(job_id, "wilor", [
            str(PYTHON), str(WORKER / "process_wilor_video.py"),
            "--input", str(source), "--hand-tracks", str(output / "hands.json"),
            "--output", str(output), "--hopformer-root", str(WILOR_ROOT),
            "--duration-seconds", f"{duration:.6f}", "--batch-size", "4", "--fast",
        ])
        update_job(job_id, progress=68)

        run(job_id, "overlay", [
            str(PYTHON), str(WORKER / "render_wilor_overlay.py"),
            "--input", str(source), "--wilor-data", str(output / "wilor-hands.json"),
            "--output", str(output / "wilor-overlay.mp4"), "--duration-seconds", f"{duration:.6f}",
        ])
        with (output / "wilor-hands.json").open("rb") as source_json, gzip.open(output / "wilor-hands.json.gzraw", "wb", compresslevel=9) as compressed:
            shutil.copyfileobj(source_json, compressed)
        update_job(job_id, progress=78)

        run(job_id, "depth", [
            str(PYTHON), str(WORKER / "create_metric_heatmap_video.py"),
            "--input", str(source), "--output", str(output / "metric-depth-overlay.mp4"),
            "--sample-fps", "2", "--width", "960",
        ])
        run(job_id, "depth", [
            str(PYTHON), str(WORKER / "create_metric_depth.py"),
            "--input", str(source), "--output", str(output),
            "--timestamp", f"{min(14.681, duration / 2):.6f}",
        ])
        update_job(job_id, progress=90)

        manifest = {
            "schema_version": "2.0", "video_id": video_id, "title": Path(job["filename"]).stem,
            "source_url": source_url, "duration_seconds": duration, "fps": fps,
            "processing": {"full_length": True, "wilor_fps": fps, "wilor_duration_seconds": duration, "depth_inference_fps": 2},
            "artifacts": {
                "video": "video-browser.mp4", "hands": "hands.json", "scene": "scene.json",
                "hand_preview": "hand-overlay-preview.jpg", "wilor_data": "wilor-hands.json.gzraw",
                "wilor_video": "wilor-overlay.mp4", "wilor_preview": "wilor-mesh-preview.jpg",
                "metric_depth_video": "metric-depth-overlay.mp4", "metric_depth_preview": "metric-depth-preview.png",
                "metric_scene": "metric-scene.json",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        run(job_id, "publish", [
            str(AWS), "s3", "sync", str(output), f"s3://{S3_BUCKET}/{derived_key}/",
            "--profile", AWS_PROFILE, "--region", AWS_REGION,
            "--cache-control", "public,max-age=3600",
        ])
        update_job(job_id, status="completed", stage="complete", progress=100,
                   message="All representations are ready", asset_base=f"{CLOUDFRONT}/{derived_key}", manifest=manifest)
    except Exception as exc:
        with (job_dir / "pipeline.log").open("a") as log:
            log.write("\n" + traceback.format_exc())
        update_job(job_id, status="failed", message=str(exc), error=str(exc))


def queue_worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            process_job(job_id)
        finally:
            job_queue.task_done()


class Handler(BaseHTTPRequestHandler):
    server_version = "Factory3DPipeline/1.0"

    def send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(200, {"ok": True, "bucket": S3_BUCKET})
            return
        if path == "/jobs":
            with jobs_lock:
                self.send_json(200, sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True))
            return
        match = re.fullmatch(r"/jobs/([a-f0-9]+)", path)
        if match and match.group(1) in jobs:
            self.send_json(200, jobs[match.group(1)])
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/jobs":
            self.send_json(404, {"error": "Not found"})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json(400, {"error": "Expected a multipart video upload"})
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
            "REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        })
        field = form["video"] if "video" in form else None
        if field is None or not getattr(field, "filename", None):
            self.send_json(400, {"error": "Choose a video file"})
            return
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(field.filename).suffix.lower() or ".mp4"
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        destination = job_dir / f"source{suffix}"
        with destination.open("wb") as target:
            shutil.copyfileobj(field.file, target, length=1024 * 1024)
        job = {
            "id": job_id, "filename": Path(field.filename).name, "local_source": str(destination),
            "size_bytes": destination.stat().st_size, "status": "queued", "stage": "queued",
            "progress": 0, "message": "Queued for processing", "created_at": time.time(),
            "stages": [{"id": key, "label": label} for key, label in STAGES],
        }
        with jobs_lock:
            jobs[job_id] = job
            write_status(job)
        job_queue.put(job_id)
        self.send_json(202, job)

    def log_message(self, format: str, *args) -> None:
        print(f"[pipeline] {self.address_string()} {format % args}")


def load_existing_jobs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    for status_path in JOBS_ROOT.glob("*/status.json"):
        try:
            job = json.loads(status_path.read_text())
            if job.get("status") == "running":
                job.update(status="interrupted", message="Restart processing by uploading the video again")
            jobs[job["id"]] = job
        except Exception:
            continue


def main() -> None:
    for required in (PYTHON, HAND_MODEL, WILOR_ROOT, AWS):
        if not required.exists():
            raise SystemExit(f"Missing required pipeline dependency: {required}")
    load_existing_jobs()
    threading.Thread(target=queue_worker, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Factory3D pipeline API listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
