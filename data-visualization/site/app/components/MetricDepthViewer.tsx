"use client";

import { Gauge, Ruler } from "lucide-react";

export function MetricDepthViewer() {
  return (
    <div className="metric-video-stage">
      <video
        src="https://d1dw8nl6ynliwf.cloudfront.net/derived/clip_322c7pdpympec/metric-depth-overlay-vda-small-2fps-full.mp4"
        poster="https://d1dw8nl6ynliwf.cloudfront.net/derived/clip_322c7pdpympec/metric-depth-preview.png"
        controls
        playsInline
        preload="metadata"
        aria-label="Video with synchronized metric-depth heat map"
      />
      <div className="metric-video-hud">
        <span><Gauge size={13} /> Metric Video Depth Anything Small</span>
        <strong>Full 5:00 heat map · 2 estimates/sec</strong>
      </div>
      <div className="depth-scale" aria-label="Depth color scale from near to far">
        <div className="depth-scale-heading"><Ruler size={13} /> Estimated camera distance</div>
        <div className="depth-gradient" />
        <div className="depth-ticks"><span>0.5 m<br />Near</span><span>2.25 m</span><span>4+ m<br />Far</span></div>
      </div>
      <div className="metric-video-disclaimer">Warm = near · Cool = far · AI estimate, not LiDAR</div>
    </div>
  );
}
