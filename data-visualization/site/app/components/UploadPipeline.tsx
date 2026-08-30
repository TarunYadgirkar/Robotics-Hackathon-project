"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, CloudUpload, LoaderCircle, Upload, X } from "lucide-react";

type PipelineJob = {
  id: string;
  filename: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  stage: string;
  progress: number;
  message: string;
  metadata?: { fps: number; duration_seconds: number; width: number; height: number };
  stages: { id: string; label: string }[];
};

const API_BASE = "/pipeline-api";

function formatDuration(seconds?: number) {
  if (!seconds) return "";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.round(seconds % 60).toString().padStart(2, "0")}`;
}

export function UploadPipeline() {
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [job, setJob] = useState<PipelineJob | null>(null);
  const [error, setError] = useState("");

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
    pollRef.current = null;
  }, []);

  const pollJob = useCallback((jobId: string) => {
    stopPolling();
    const refresh = async () => {
      try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`);
        if (!response.ok) throw new Error("Processing service is unavailable");
        const next = await response.json() as PipelineJob;
        setJob(next);
        if (["completed", "failed", "interrupted"].includes(next.status)) stopPolling();
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Could not read processing status");
        stopPolling();
      }
    };
    void refresh();
    pollRef.current = window.setInterval(refresh, 2000);
  }, [stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  const upload = (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|avi|webm)$/i.test(file.name)) {
      setError("Choose an MP4, MOV, M4V, AVI, or WebM video.");
      return;
    }
    setError("");
    setJob(null);
    setUploadProgress(0);
    const body = new FormData();
    body.append("video", file);
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/jobs`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setUploadProgress(Math.round((event.loaded / event.total) * 100));
    };
    request.onerror = () => setError("The local processing service is not running.");
    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        setError("The video could not be queued.");
        return;
      }
      const created = JSON.parse(request.responseText) as PipelineJob;
      setJob(created);
      setUploadProgress(100);
      pollJob(created.id);
    };
    request.send(body);
  };

  const activeStageIndex = job?.stages.findIndex((stage) => stage.id === job.stage) ?? -1;

  return (
    <section className="upload-pipeline" aria-label="Upload and process video">
      <div className="upload-heading">
        <div><CloudUpload size={16} /><span>Process a video</span></div>
        {job && <button onClick={() => { stopPolling(); setJob(null); setUploadProgress(0); }} aria-label="Clear upload"><X size={14} /></button>}
      </div>
      {!job && (
        <button
          className={`video-dropzone ${dragging ? "dragging" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files[0]); }}
        >
          <Upload size={18} /><strong>Drop video here</strong><span>or choose a file</span>
          <input ref={inputRef} type="file" accept="video/*,.mp4,.mov,.m4v,.avi,.webm" onChange={(event) => upload(event.target.files?.[0])} />
        </button>
      )}
      {!job && uploadProgress > 0 && uploadProgress < 100 && <div className="upload-transfer"><span style={{ width: `${uploadProgress}%` }} /></div>}
      {job && (
        <div className={`pipeline-job ${job.status}`}>
          <div className="pipeline-file"><strong>{job.filename}</strong><span>{job.metadata ? `${formatDuration(job.metadata.duration_seconds)} · ${job.metadata.fps.toFixed(2)} FPS` : "Inspecting video"}</span></div>
          <div className="pipeline-progress"><span style={{ width: `${job.progress}%` }} /></div>
          <div className="pipeline-status"><span>{job.message}</span><strong>{job.progress}%</strong></div>
          <div className="pipeline-stages">
            {job.stages.map((stage, index) => {
              const complete = job.status === "completed" || index < activeStageIndex;
              const active = index === activeStageIndex && job.status === "running";
              return <div className={complete ? "complete" : active ? "active" : ""} key={stage.id}><i>{complete ? <Check size={10} /> : active ? <LoaderCircle size={10} /> : null}</i><span>{stage.label}</span></div>;
            })}
          </div>
          {job.status === "completed" && <p className="pipeline-complete">Full-length outputs are stored in S3. WiLoR matches the source duration and FPS.</p>}
        </div>
      )}
      {error && <p className="upload-error">{error}</p>}
    </section>
  );
}
