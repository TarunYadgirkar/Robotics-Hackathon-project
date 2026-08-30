"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Pause, Play, RotateCcw } from "lucide-react";

type SceneData = {
  method: string;
  point_count: number;
  source_size: { width: number; height: number };
  points: [number, number, number, number, number, number][];
};

const SAMPLE_START = 145;
const SAMPLE_END = 165;
const GRID_STRIDE = 5;

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

export function PointCloudViewer() {
  const hostRef = useRef<HTMLDivElement>(null);
  const resetRef = useRef<(() => void) | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [status, setStatus] = useState("Building live depth surface…");
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(SAMPLE_START);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.FogExp2(0x000000, 0.075);

    const camera = new THREE.PerspectiveCamera(58, 1, 0.01, 40);
    camera.position.set(0, 0.05, 5.3);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, -1.9);
    controls.minDistance = 1.1;
    controls.maxDistance = 9;
    controls.maxPolarAngle = Math.PI * 0.82;

    const video = document.createElement("video");
    video.src = `/video-stream/sample.mp4#t=${SAMPLE_START}`;
    video.preload = "auto";
    video.muted = true;
    video.loop = false;
    video.playsInline = true;
    video.crossOrigin = "anonymous";
    videoRef.current = video;

    const handleMetadata = () => {
      video.currentTime = SAMPLE_START;
      setCurrentTime(SAMPLE_START);
    };
    const handleTimeUpdate = () => {
      if (video.currentTime >= SAMPLE_END) {
        video.currentTime = SAMPLE_START;
      }
      setCurrentTime(video.currentTime);
    };
    const handlePlay = () => setPlaying(true);
    const handlePause = () => setPlaying(false);
    video.addEventListener("loadedmetadata", handleMetadata);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);
    video.load();

    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.colorSpace = THREE.SRGBColorSpace;
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    videoTexture.generateMipmaps = false;

    const reset = () => {
      camera.position.set(0, 0.05, 5.3);
      controls.target.set(0, 0, -1.9);
      controls.update();
    };
    resetRef.current = reset;

    let surface: THREE.Mesh | null = null;
    let disposed = false;
    fetch("/data/clip_322c7pdpympec/scene.json")
      .then((response) => {
        if (!response.ok) throw new Error("Scene artifact was not found");
        return response.json() as Promise<SceneData>;
      })
      .then((data) => {
        if (disposed) return;
        const columns = Math.ceil(data.source_size.width / GRID_STRIDE);
        const rows = Math.ceil(data.source_size.height / GRID_STRIDE);
        const positions = new Float32Array(data.points.length * 3);
        const uvs = new Float32Array(data.points.length * 2);

        data.points.forEach((point, index) => {
          positions.set(point.slice(0, 3), index * 3);
          const row = Math.floor(index / columns);
          const column = index % columns;
          uvs.set([column / (columns - 1), 1 - row / (rows - 1)], index * 2);
        });

        const indices: number[] = [];
        const addTriangle = (a: number, b: number, c: number) => {
          const depths = [data.points[a][2], data.points[b][2], data.points[c][2]];
          if (Math.max(...depths) - Math.min(...depths) < 0.52) indices.push(a, b, c);
        };
        for (let row = 0; row < rows - 1; row += 1) {
          for (let column = 0; column < columns - 1; column += 1) {
            const topLeft = row * columns + column;
            const topRight = topLeft + 1;
            const bottomLeft = topLeft + columns;
            const bottomRight = bottomLeft + 1;
            addTriangle(topLeft, bottomLeft, topRight);
            addTriangle(topRight, bottomLeft, bottomRight);
          }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();

        const material = new THREE.MeshBasicMaterial({
          map: videoTexture,
          side: THREE.DoubleSide,
          toneMapped: false,
        });
        surface = new THREE.Mesh(geometry, material);
        scene.add(surface);
        setStatus(`${data.point_count.toLocaleString()} depth vertices · live video surface`);
      })
      .catch((error: Error) => setStatus(error.message));

    const resize = () => {
      const { clientWidth, clientHeight } = host;
      if (!clientWidth || !clientHeight) return;
      renderer.setSize(clientWidth, clientHeight, false);
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    let animationFrame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    };
    render();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      video.pause();
      video.removeAttribute("src");
      video.load();
      video.removeEventListener("loadedmetadata", handleMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
      videoRef.current = null;
      surface?.geometry.dispose();
      if (surface) (surface.material as THREE.Material).dispose();
      videoTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      if (video.currentTime < SAMPLE_START || video.currentTime >= SAMPLE_END) video.currentTime = SAMPLE_START;
      await video.play();
    } else {
      video.pause();
    }
  };

  return (
    <div className="cloud-shell">
      <div className="cloud-stage" ref={hostRef} aria-label="Interactive depth-shaped video reconstruction" />
      <div className="fog-vignette" aria-hidden="true" />
      <div className="fog-bank fog-bank-left" aria-hidden="true" />
      <div className="fog-bank fog-bank-right" aria-hidden="true" />
      <div className="cloud-hud">
        <span className="live-dot" />
        {status}
      </div>
      <button className="spatial-play" onClick={togglePlayback} aria-label={playing ? "Pause video in 3D" : "Play video in 3D"}>
        {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
        {playing ? "Pause 3D video" : "Play in 3D"}
      </button>
      <div className="spatial-time">{formatTime(currentTime)} — {formatTime(SAMPLE_END)}</div>
      <div className="axis-key" aria-hidden="true">
        <span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z</span>
      </div>
      <button className="reset-view" onClick={() => resetRef.current?.()} aria-label="Reset 3D camera">
        <RotateCcw size={15} /> Reset view
      </button>
      <div className="unknown-space"><span /> Black fog marks unobserved space</div>
      <div className="drag-hint">Drag to orbit · Scroll to move through depth</div>
    </div>
  );
}
