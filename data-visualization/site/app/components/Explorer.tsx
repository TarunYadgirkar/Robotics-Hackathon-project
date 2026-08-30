"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  ChevronRight,
  CircleDot,
  Hand,
  Layers3,
  Play,
  Ruler,
  Search,
  Sparkles,
} from "lucide-react";
import { PointCloudViewer } from "./PointCloudViewer";
import { MetricDepthViewer } from "./MetricDepthViewer";
import { Hand3DViewer } from "./Hand3DViewer";
import { UploadPipeline } from "./UploadPipeline";

type Landmark = { x: number; y: number; z: number };
type HandRecord = { handedness: string; confidence: number; landmarks: Landmark[] };
type HandFrame = { timestamp_seconds: number; hands: HandRecord[] };
type HandData = {
  connections: [number, number][];
  frames: HandFrame[];
  summary: { sampled_frames: number; frames_with_hands: number; detection_coverage: number; maximum_hands: number };
};

const ASSET_BASE = "https://d1dw8nl6ynliwf.cloudfront.net/derived/clip_322c7pdpympec";
const DATA_BASE = import.meta.env.DEV ? "/s3-derived" : ASSET_BASE;
const VIDEO_URL = `${ASSET_BASE}/video-18fps-5min.mp4?v=1`;
const SAMPLE_START = 0;
const SAMPLE_END = 300;

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
  const [layout, setLayout] = useState<"stacked" | "tabs">("stacked");
  const [mode, setMode] = useState<"video" | "wilor" | "hands3d" | "map" | "metric">("video");
  const [showJoints, setShowJoints] = useState(true);
  const [currentTime, setCurrentTime] = useState(SAMPLE_START);
  const [handData, setHandData] = useState<HandData | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch(`${DATA_BASE}/hands.json?v=tracked-2997`)
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
      <aside className="sidebar">
        <UploadPipeline />
        <div className="library-heading">
          <span>Video library</span>
        </div>
        <label className="search-box"><Search size={15} /><input placeholder="Search 424 videos" aria-label="Search videos" /></label>
        <div className="collection-label"><span>Manufacturing tasks</span><span>50</span></div>
        <div className="video-card active">
          <div className="thumbnail">
            <img src={`${ASSET_BASE}/hand-overlay-preview.jpg`} alt="Axle shaft cutting workstation" />
            <span className="duration">5:00</span>
          </div>
          <div className="video-card-copy">
            <strong>Axle shaft cutting</strong>
            <span>clip_322c7pdpympec</span>
            <div><span className="ready-badge">Full joints ready</span><span>18 FPS</span></div>
          </div>
        </div>
        <div className="queued-card">
          <div className="queue-icon"><Box size={17} /></div>
          <div><strong>423 more videos</strong><span>Waiting for batch pipeline</span></div>
          <ChevronRight size={15} />
        </div>
        <div className="dataset-summary">
          <span className="eyebrow">Dataset coverage</span>
          <div><strong>1</strong><span>/ 424 processed</span></div>
          <div className="progress-track"><span /></div>
          <p>Processed assets available in the local workspace.</p>
        </div>
      </aside>

      <section className="workspace">
        <div className="workspace-heading">
          <div>
            <div className="breadcrumb">VIDEOS <ChevronRight size={12} /> AXLE SHAFT CUTTING</div>
            <h1>Axle shaft cutting</h1>
          </div>
          <div className="layout-toggle" role="group" aria-label="Analysis layout">
            <button className={layout === "stacked" ? "active" : ""} onClick={() => setLayout("stacked")}>All views</button>
            <button className={layout === "tabs" ? "active" : ""} onClick={() => setLayout("tabs")}>Single view</button>
          </div>
        </div>

        {layout === "tabs" && (
          <div className="mode-switch" role="tablist" aria-label="Viewer mode">
            <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")}><Play size={14} /> Video + joints</button>
            <button className={mode === "wilor" ? "active" : ""} onClick={() => setMode("wilor")}><Hand size={14} /> WiLoR mesh</button>
            <button className={mode === "hands3d" ? "active" : ""} onClick={() => setMode("hands3d")}><Box size={14} /> 3D hands</button>
            <button className={mode === "map" ? "active" : ""} onClick={() => setMode("map")}><Layers3 size={14} /> 3D map</button>
            <button className={mode === "metric" ? "active" : ""} onClick={() => setMode("metric")}><Ruler size={14} /> Depth heat map</button>
          </div>
        )}

        <div className={`analysis-stack ${layout}`}>
          <section className="analysis-block" hidden={layout === "tabs" && mode !== "video"}>
            <div className="analysis-heading">
              <div><span>01</span><h2>Video + joints</h2></div>
              <p>POV workstation capture · 1920 × 1080 · 18 fps · full 5-minute joint analysis</p>
            </div>
            <div className="viewer-grid">
              <div className="viewer-card">
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
                  <span className="source-tag"><CircleDot size={13} /> 29.97 FPS source tracking</span>
                  <span className={`tracking-tag ${activeFrame ? "tracking" : ""}`}>
                    <Hand size={13} /> {activeFrame ? `${activeFrame.hands.length} hands tracked` : "Outside analyzed segment"}
                  </span>
                </div>
                {!playing && (
                  <button className="center-play" onClick={() => void videoRef.current?.play().catch(() => undefined)} aria-label="Play video"><Play size={23} fill="currentColor" /></button>
                )}
              </div>
                <div className="viewer-toolbar">
                  <button className={`layer-toggle ${showJoints ? "on" : ""}`} onClick={() => setShowJoints((value) => !value)}>
                    <span /><Hand size={15} /> Hand joints
                  </button>
                  <div className="timeline-status">
                    <span>{formatTime(currentTime)}</span>
                    <div className="mini-track"><span style={{ width: `${Math.min(100, Math.max(0, ((currentTime - SAMPLE_START) / (SAMPLE_END - SAMPLE_START)) * 100))}%` }} /></div>
                    <span>{formatTime(SAMPLE_END)}</span>
                  </div>
                  <button className="proof-jump" onClick={seekToProof}>Restart analyzed video</button>
                </div>
              </div>
              <aside className="inspector">
                <div className="inspector-section metrics-section">
                  <div className="section-title"><span>Hand analysis</span><Hand size={15} /></div>
                  <div className="metric-grid">
                    <div><strong>95.9%</strong><span>Detection coverage</span></div>
                    <div><strong>5,400</strong><span>Frames analyzed</span></div>
                    <div><strong>21</strong><span>Joints per hand</span></div>
                    <div><strong>91.5%</strong><span>Both-hands continuity</span></div>
                  </div>
                  <div className="hand-legend"><span><i className="left-hand" /> Left hand</span><span><i className="right-hand" /> Right hand</span></div>
                </div>
              </aside>
            </div>
          </section>

          <section className="analysis-block" hidden={layout === "tabs" && mode !== "wilor"}>
            <div className="analysis-heading">
              <div><span>02</span><h2>WiLoR mesh</h2></div>
              <p>WiLoR MANO reconstruction · 1920 × 1080 · 18 fps · 60-second mesh analysis</p>
            </div>
            <div className="viewer-grid">
              <div className="viewer-card">
              <div className="wilor-stage">
                <video
                  src={`${ASSET_BASE}/wilor-overlay-60s.mp4?v=5`}
                  poster={`${ASSET_BASE}/wilor-mesh-preview-60s.jpg`}
                  controls
                  preload="metadata"
                  playsInline
                  aria-label="One-minute WiLoR MANO hand mesh reconstruction"
                />
              </div>
                <div className="viewer-toolbar"><div className="map-note wilor-note"><Hand size={15} /> Frame-matched: 1,080 inference samples · 1,080 rendered frames</div></div>
              </div>
              <aside className="inspector">
                <div className="inspector-section metrics-section">
                  <div className="section-title"><span>WiLoR mesh analysis</span><Hand size={15} /></div>
                  <div className="metric-grid">
                    <div><strong>2,131</strong><span>Hand meshes</span></div>
                    <div><strong>1,080</strong><span>Frames analyzed</span></div>
                    <div><strong>778</strong><span>Vertices per hand</span></div>
                    <div><strong>21</strong><span>Joints per hand</span></div>
                  </div>
                </div>
              </aside>
            </div>
          </section>

          <section className="analysis-block" hidden={layout === "tabs" && mode !== "hands3d"}>
            <div className="analysis-heading">
              <div><span>03</span><h2>Interactive 3D hands</h2></div>
              <p>Interactive MANO geometry · 778 vertices per hand · 60-second synchronized segment</p>
            </div>
            <div className="viewer-grid">
              <div className="viewer-card">
                <Hand3DViewer />
                <div className="viewer-toolbar"><div className="map-note hands3d-note"><Box size={15} /> Video-synchronized WiLoR meshes · orbitable model-relative 3D</div></div>
              </div>
              <aside className="inspector">
                <div className="inspector-section metrics-section">
                  <div className="section-title"><span>Interactive 3D hands</span><Box size={15} /></div>
                  <div className="metric-grid">
                    <div><strong>2</strong><span>Maximum hands</span></div>
                    <div><strong>18 FPS</strong><span>Playback rate</span></div>
                    <div><strong>778</strong><span>Mesh vertices</span></div>
                    <div><strong>1,538</strong><span>Mesh faces</span></div>
                  </div>
                  <p className="inspector-copy">Orbit, zoom, and inspect the synchronized MANO hand geometry independently from the camera view.</p>
                </div>
              </aside>
            </div>
          </section>

          <section className="analysis-block" hidden={layout === "tabs" && mode !== "map"}>
            <div className="analysis-heading">
              <div><span>04</span><h2>3D video surface</h2></div>
              <p>Relative-depth video surface · 9,216 vertices · 20-second reconstructed sample</p>
            </div>
            <div className="viewer-grid">
              <div className="viewer-card">
                <PointCloudViewer />
                <div className="viewer-toolbar"><div className="map-note"><Sparkles size={15} /> Live footage on relative-depth geometry · black fog is unobserved space</div></div>
              </div>
              <aside className="inspector">
                <div className="inspector-section mapping-section">
                  <div className="section-title"><span>3D representation</span><Layers3 size={15} /></div>
                  <div className="point-count"><strong>9,216</strong><span>depth-shaped video vertices</span></div>
                  <p>The footage is projected onto reconstructed relative-depth geometry. Black regions represent unobserved space.</p>
                </div>
              </aside>
            </div>
          </section>

          <section className="analysis-block" hidden={layout === "tabs" && mode !== "metric"}>
            <div className="analysis-heading">
              <div><span>05</span><h2>Depth heat map</h2></div>
              <p>Apple Depth Pro estimate · 1920 × 1080 · 20-second synchronized sample</p>
            </div>
            <div className="viewer-grid">
              <div className="viewer-card">
                <MetricDepthViewer />
                <div className="viewer-toolbar"><div className="map-note metric-note"><Ruler size={15} /> Synchronized metric-depth heat map · warm is near, cool is far</div></div>
              </div>
              <aside className="inspector">
                <div className="inspector-section metrics-section">
                  <div className="section-title"><span>Metric depth</span><Ruler size={15} /></div>
                  <div className="metric-grid">
                    <div><strong>1.87 m</strong><span>Median depth</span></div>
                    <div><strong>9,216</strong><span>Depth samples</span></div>
                    <div><strong>70°</strong><span>Estimated HFOV</span></div>
                    <div><strong>Depth Pro</strong><span>Estimation model</span></div>
                  </div>
                </div>
              </aside>
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
