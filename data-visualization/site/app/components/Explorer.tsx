"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Database,
  Hand,
  Layers3,
  Play,
  Search,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { PointCloudViewer } from "./PointCloudViewer";

type Landmark = { x: number; y: number; z: number };
type HandRecord = { handedness: string; confidence: number; landmarks: Landmark[] };
type HandFrame = { timestamp_seconds: number; hands: HandRecord[] };
type HandData = {
  connections: [number, number][];
  frames: HandFrame[];
  summary: { sampled_frames: number; frames_with_hands: number; detection_coverage: number; maximum_hands: number };
};

const VIDEO_URL = "https://d1dw8nl6ynliwf.cloudfront.net/videos/axle-shaft-cutting/clip_322c7pdpympec.mp4";
const SAMPLE_START = 145;
const SAMPLE_END = 165;

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function HandOverlay({ frame, connections }: { frame?: HandFrame; connections: [number, number][] }) {
  if (!frame) return null;
  return (
    <svg className="joint-overlay" viewBox="0 0 1000 562.5" preserveAspectRatio="none" aria-label="Detected hand joints">
      {frame.hands.map((hand, handIndex) => {
        const color = hand.handedness === "Left" ? "#7CFF8D" : "#3DDCFF";
        return (
          <g key={`${hand.handedness}-${handIndex}`}>
            {connections.map(([start, end]) => (
              <line
                key={`${start}-${end}`}
                x1={hand.landmarks[start].x * 1000}
                y1={hand.landmarks[start].y * 562.5}
                x2={hand.landmarks[end].x * 1000}
                y2={hand.landmarks[end].y * 562.5}
                stroke={color}
                strokeWidth="3.5"
                strokeLinecap="round"
              />
            ))}
            {hand.landmarks.map((point, index) => (
              <circle
                key={index}
                cx={point.x * 1000}
                cy={point.y * 562.5}
                r={index === 0 ? 6 : 4.2}
                fill="#F7FBF8"
                stroke={color}
                strokeWidth="2.4"
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

export function Explorer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const initialSeekDone = useRef(false);
  const [mode, setMode] = useState<"video" | "map">("video");
  const [showJoints, setShowJoints] = useState(true);
  const [currentTime, setCurrentTime] = useState(SAMPLE_START);
  const [handData, setHandData] = useState<HandData | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch("/data/clip_322c7pdpympec/hands.json")
      .then((response) => response.json() as Promise<HandData>)
      .then(setHandData)
      .catch(() => setHandData(null));
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const initializePosition = () => {
      if (initialSeekDone.current) return;
      initialSeekDone.current = true;
      video.currentTime = SAMPLE_START;
      setCurrentTime(SAMPLE_START);
    };
    if (video.readyState >= 1) initializePosition();
    else video.addEventListener("loadedmetadata", initializePosition, { once: true });
    return () => video.removeEventListener("loadedmetadata", initializePosition);
  }, []);

  const activeFrame = useMemo(() => {
    if (!handData || currentTime < SAMPLE_START - 0.25 || currentTime > SAMPLE_END + 0.25) return undefined;
    return handData.frames.reduce((nearest, frame) =>
      Math.abs(frame.timestamp_seconds - currentTime) < Math.abs(nearest.timestamp_seconds - currentTime)
        ? frame
        : nearest
    );
  }, [currentTime, handData]);

  const seekToProof = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = SAMPLE_START;
    setCurrentTime(SAMPLE_START);
  };

  const prepareProofPosition = () => {
    if (!videoRef.current) return;
    if (videoRef.current.currentTime < SAMPLE_START - 1 || videoRef.current.currentTime > SAMPLE_END + 1) {
      videoRef.current.currentTime = SAMPLE_START;
    }
    setCurrentTime(videoRef.current.currentTime || SAMPLE_START);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Layers3 size={18} /></div>
          <div><strong>Factory Atlas</strong><span>Spatial operations intelligence</span></div>
        </div>
        <div className="topbar-center">
          <span className="environment-pill"><span /> One-video validation</span>
          <span className="cloud-source">S3 + CloudFront source</span>
        </div>
        <button className="dataset-button"><Database size={15} /> Dataset <ChevronDown size={14} /></button>
      </header>

      <aside className="sidebar">
        <div className="library-heading">
          <span>Video library</span>
          <button aria-label="Filter videos"><SlidersHorizontal size={16} /></button>
        </div>
        <label className="search-box"><Search size={15} /><input placeholder="Search 424 videos" aria-label="Search videos" /></label>
        <div className="collection-label"><span>Manufacturing tasks</span><span>50</span></div>
        <button className="video-card active" onClick={() => setMode("video")}>
          <div className="thumbnail">
            <img src="/data/clip_322c7pdpympec/hand-overlay-preview.jpg" alt="Axle shaft cutting workstation" />
            <span className="duration">5:00</span>
          </div>
          <div className="video-card-copy">
            <strong>Axle shaft cutting</strong>
            <span>clip_322c7pdpympec</span>
            <div><span className="ready-badge">Preview ready</span><span>2 hands</span></div>
          </div>
        </button>
        <div className="queued-card">
          <div className="queue-icon"><Box size={17} /></div>
          <div><strong>423 more videos</strong><span>Waiting for batch pipeline</span></div>
          <ChevronRight size={15} />
        </div>
        <div className="dataset-summary">
          <span className="eyebrow">Dataset coverage</span>
          <div><strong>1</strong><span>/ 424 processed</span></div>
          <div className="progress-track"><span /></div>
          <p>One video is intentionally enabled for this validation.</p>
        </div>
      </aside>

      <section className="workspace">
        <div className="workspace-heading">
          <div>
            <div className="breadcrumb">VIDEOS <ChevronRight size={12} /> AXLE SHAFT CUTTING</div>
            <h1>Axle shaft cutting</h1>
            <p>POV workstation capture · 1920 × 1080 · 29.97 fps</p>
          </div>
          <div className="mode-switch" role="tablist" aria-label="Viewer mode">
            <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")}><Play size={14} /> Video + joints</button>
            <button className={mode === "map" ? "active" : ""} onClick={() => setMode("map")}><Layers3 size={14} /> 3D map</button>
          </div>
        </div>

        <div className="viewer-grid">
          <div className="viewer-card">
            {mode === "video" ? (
              <div className="video-stage">
                <video
                  ref={videoRef}
                  src={`${VIDEO_URL}#t=${SAMPLE_START}`}
                  controls
                  preload="metadata"
                  playsInline
                  onLoadedMetadata={prepareProofPosition}
                  onCanPlay={prepareProofPosition}
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                />
                {showJoints && <HandOverlay frame={activeFrame} connections={handData?.connections ?? []} />}
                <div className="video-top-hud">
                  <span className="source-tag"><CircleDot size={13} /> CloudFront source</span>
                  <span className={`tracking-tag ${activeFrame ? "tracking" : ""}`}>
                    <Hand size={13} /> {activeFrame ? `${activeFrame.hands.length} hands tracked` : "Outside analyzed segment"}
                  </span>
                </div>
                {!playing && (
                  <button className="center-play" onClick={() => videoRef.current?.play()} aria-label="Play video"><Play size={23} fill="currentColor" /></button>
                )}
              </div>
            ) : <PointCloudViewer />}
            <div className="viewer-toolbar">
              {mode === "video" ? (
                <>
                  <button className={`layer-toggle ${showJoints ? "on" : ""}`} onClick={() => setShowJoints((value) => !value)}>
                    <span /><Hand size={15} /> Hand joints
                  </button>
                  <div className="timeline-status">
                    <span>{formatTime(currentTime)}</span>
                    <div className="mini-track"><span style={{ width: `${Math.min(100, Math.max(0, ((currentTime - SAMPLE_START) / 20) * 100))}%` }} /></div>
                    <span>{formatTime(SAMPLE_END)}</span>
                  </div>
                  <button className="proof-jump" onClick={seekToProof}>Jump to analyzed segment</button>
                </>
              ) : (
                <div className="map-note"><Sparkles size={15} /> Live footage on relative-depth geometry · black fog is unobserved space</div>
              )}
            </div>
          </div>

          <aside className="inspector">
            <div className="inspector-section status-section">
              <div className="section-title"><span>Proof status</span><span className="preview-chip">PREVIEW</span></div>
              <div className="status-row"><span>Hand mapping</span><strong><i className="status-ok" /> Ready</strong></div>
              <div className="status-row"><span>3D video surface</span><strong><i className="status-ok" /> Ready</strong></div>
              <div className="status-row"><span>Gaussian splat</span><strong className="muted">GPU pending</strong></div>
            </div>
            <div className="inspector-section metrics-section">
              <div className="section-title"><span>Hand analysis</span><Hand size={15} /></div>
              <div className="metric-grid">
                <div><strong>100%</strong><span>Detection coverage</span></div>
                <div><strong>61</strong><span>Frames analyzed</span></div>
                <div><strong>21</strong><span>Joints per hand</span></div>
                <div><strong>2</strong><span>Maximum hands</span></div>
              </div>
              <div className="hand-legend"><span><i className="left-hand" /> Left hand</span><span><i className="right-hand" /> Right hand</span></div>
            </div>
            <div className="inspector-section mapping-section">
              <div className="section-title"><span>3D representation</span><Layers3 size={15} /></div>
              <div className="point-count"><strong>9,216</strong><span>depth-shaped video vertices</span></div>
              <p>The real footage plays across reconstructed depth. Black fog identifies space the camera never observed.</p>
              <button onClick={() => setMode("map")}>Open spatial view <ChevronRight size={15} /></button>
            </div>
            <div className="quality-callout">
              <span className="eyebrow">Capture finding</span>
              <strong>Limited camera translation</strong>
              <p>The POV remains mostly fixed. Depth and hands are usable; full free-view reconstruction will require validation on GPU.</p>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}
