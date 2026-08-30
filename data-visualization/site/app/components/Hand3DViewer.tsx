"use client";

import { useEffect, useRef, useState } from "react";
import { Eye, EyeOff, RotateCcw } from "lucide-react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Vector3Tuple = [number, number, number];
type Vector2Tuple = [number, number];
type HandMeshRecord = {
  timestamp_seconds: number;
  handedness: "Left" | "Right";
  camera_translation: Vector3Tuple;
  joints: Vector3Tuple[];
  joints_2d: Vector2Tuple[];
  vertices: Vector3Tuple[];
};
type WiLoRData = {
  sample_fps: number;
  mesh: { faces: [number, number, number][] };
  hands: HandMeshRecord[];
};
type HandFrame = { timestamp: number; hands: HandMeshRecord[] };
type HandVisual = {
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
  joints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
  bones: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
};

const ASSET_BASE = "https://d1dw8nl6ynliwf.cloudfront.net/derived/clip_322c7pdpympec";
const DATA_BASE = import.meta.env.DEV ? "/s3-derived" : ASSET_BASE;

const VIDEO_WIDTH = 1920;
const VIDEO_HEIGHT = 1080;
const HAND_SCALE = 10;
const MIN_HAND_SEPARATION = 1.75;
const DEPTH_CENTER = 25.5;
const DEPTH_SCALE = 0.24;
const POSE_SMOOTHING = 0.28;
const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function formatTime(seconds: number) {
  return `0:${Math.floor(seconds).toString().padStart(2, "0")}`;
}

export function Hand3DViewer() {
  const hostRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const resetRef = useRef<(() => void) | null>(null);
  const [showVideo, setShowVideo] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [meshStatus, setMeshStatus] = useState<"loading" | "ready" | "error">("loading");
  const [meshError, setMeshError] = useState("");

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let stopped = false;
    let animationFrame = 0;
    let data: WiLoRData | null = null;
    let frames: HandFrame[] = [];
    let lastFrameIndex = -1;
    let lastVideoTime: number | undefined;
    const smoothedDepth: Record<"Left" | "Right", number | undefined> = { Left: undefined, Right: undefined };
    const smoothedPose: Record<"Left" | "Right", { vertices?: Float32Array; joints?: Float32Array }> = {
      Left: {},
      Right: {},
    };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020504);
    scene.fog = new THREE.FogExp2(0x020504, 0.075);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    const defaultCamera = new THREE.Vector3(0, 0.1, 5.2);
    camera.position.copy(defaultCamera);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 1.3;
    controls.maxDistance = 12;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0xbcecff, 0x172019, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(3, 5, 5);
    scene.add(keyLight);

    const grid = new THREE.GridHelper(12, 24, 0x315044, 0x14221c);
    grid.position.y = -1.35;
    scene.add(grid);

    const visualGroup = new THREE.Group();
    scene.add(visualGroup);

    const makeVisual = (color: number): HandVisual => {
      const meshGeometry = new THREE.BufferGeometry();
      meshGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(778 * 3), 3));
      const mesh = new THREE.Mesh(
        meshGeometry,
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.12,
          metalness: 0.05,
          roughness: 0.48,
          transparent: true,
          opacity: 0.72,
          side: THREE.DoubleSide,
        })
      );

      const jointGeometry = new THREE.BufferGeometry();
      jointGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(21 * 3), 3));
      const joints = new THREE.Points(
        jointGeometry,
        new THREE.PointsMaterial({ color: 0xffffff, size: 0.055, sizeAttenuation: true })
      );

      const boneGeometry = new THREE.BufferGeometry();
      boneGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(CONNECTIONS.length * 6), 3));
      const bones = new THREE.LineSegments(
        boneGeometry,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 })
      );

      mesh.visible = false;
      joints.visible = false;
      bones.visible = false;
      visualGroup.add(mesh, joints, bones);
      return { mesh, joints, bones };
    };

    const visuals = [makeVisual(0x7cff8d), makeVisual(0x42dfff)];

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    resize();

    resetRef.current = () => {
      camera.position.copy(defaultCamera);
      controls.target.set(0, 0, 0);
      controls.update();
    };

    const imageCenterX = (hand: HandMeshRecord) =>
      hand.joints_2d.reduce((sum, point) => sum + point[0], 0) / hand.joints_2d.length;

    const placeHand = (visual: HandVisual, hand: HandMeshRecord, separationShift = 0) => {
      const handColor = new THREE.Color(hand.handedness === "Left" ? 0x7cff8d : 0x42dfff);
      visual.mesh.material.color.copy(handColor);
      visual.mesh.material.emissive.copy(handColor);
      visual.bones.material.color.copy(handColor);
      const wrist = hand.joints[0];
      const center = hand.joints_2d.reduce(
        (sum, point) => [sum[0] + point[0], sum[1] + point[1]],
        [0, 0]
      );
      const anchorX = (center[0] / hand.joints_2d.length / VIDEO_WIDTH - 0.5) * 4.4 + separationShift;
      const anchorY = -(center[1] / hand.joints_2d.length / VIDEO_HEIGHT - 0.54) * 2.5;
      const targetDepth = -THREE.MathUtils.clamp(hand.camera_translation[2] - DEPTH_CENTER, -5, 5) * DEPTH_SCALE;
      const previousDepth = smoothedDepth[hand.handedness];
      const anchorZ = previousDepth === undefined
        ? targetDepth
        : THREE.MathUtils.lerp(previousDepth, targetDepth, 0.32);
      smoothedDepth[hand.handedness] = anchorZ;

      const transform = (point: Vector3Tuple) => [
        (point[0] - wrist[0]) * HAND_SCALE + anchorX,
        -(point[1] - wrist[1]) * HAND_SCALE + anchorY,
        -(point[2] - wrist[2]) * HAND_SCALE + anchorZ,
      ] as Vector3Tuple;

      const meshPositions = visual.mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
      const pose = smoothedPose[hand.handedness];
      const initializeVertices = !pose.vertices;
      if (!pose.vertices) pose.vertices = new Float32Array(hand.vertices.length * 3);
      hand.vertices.forEach((vertex, index) => {
        const target = transform(vertex);
        const offset = index * 3;
        if (initializeVertices) {
          pose.vertices![offset] = target[0];
          pose.vertices![offset + 1] = target[1];
          pose.vertices![offset + 2] = target[2];
        } else {
          pose.vertices![offset] = THREE.MathUtils.lerp(pose.vertices![offset], target[0], POSE_SMOOTHING);
          pose.vertices![offset + 1] = THREE.MathUtils.lerp(pose.vertices![offset + 1], target[1], POSE_SMOOTHING);
          pose.vertices![offset + 2] = THREE.MathUtils.lerp(pose.vertices![offset + 2], target[2], POSE_SMOOTHING);
        }
        meshPositions.setXYZ(index, pose.vertices![offset], pose.vertices![offset + 1], pose.vertices![offset + 2]);
      });
      meshPositions.needsUpdate = true;
      visual.mesh.geometry.computeVertexNormals();

      const initializeJoints = !pose.joints;
      if (!pose.joints) pose.joints = new Float32Array(hand.joints.length * 3);
      const transformedJoints = hand.joints.map((joint, index) => {
        const target = transform(joint);
        const offset = index * 3;
        if (initializeJoints) {
          pose.joints![offset] = target[0];
          pose.joints![offset + 1] = target[1];
          pose.joints![offset + 2] = target[2];
        } else {
          pose.joints![offset] = THREE.MathUtils.lerp(pose.joints![offset], target[0], POSE_SMOOTHING);
          pose.joints![offset + 1] = THREE.MathUtils.lerp(pose.joints![offset + 1], target[1], POSE_SMOOTHING);
          pose.joints![offset + 2] = THREE.MathUtils.lerp(pose.joints![offset + 2], target[2], POSE_SMOOTHING);
        }
        return [pose.joints![offset], pose.joints![offset + 1], pose.joints![offset + 2]] as Vector3Tuple;
      });
      const jointPositions = visual.joints.geometry.getAttribute("position") as THREE.BufferAttribute;
      transformedJoints.forEach((joint, index) => jointPositions.setXYZ(index, ...joint));
      jointPositions.needsUpdate = true;

      const bonePositions = visual.bones.geometry.getAttribute("position") as THREE.BufferAttribute;
      CONNECTIONS.forEach(([start, end], index) => {
        bonePositions.setXYZ(index * 2, ...transformedJoints[start]);
        bonePositions.setXYZ(index * 2 + 1, ...transformedJoints[end]);
      });
      bonePositions.needsUpdate = true;
      visual.mesh.visible = true;
      visual.joints.visible = true;
      visual.bones.visible = true;
    };

    const updateFrame = () => {
      if (!data || frames.length === 0) return;
      const time = videoRef.current?.currentTime ?? 0;
      if (lastVideoTime !== undefined && Math.abs(time - lastVideoTime) > 0.35) {
        smoothedPose.Left = {};
        smoothedPose.Right = {};
        smoothedDepth.Left = undefined;
        smoothedDepth.Right = undefined;
        visuals.forEach((visual) => {
          visual.mesh.visible = false;
          visual.joints.visible = false;
          visual.bones.visible = false;
        });
      }
      lastVideoTime = time;
      const frameIndex = Math.max(0, Math.min(frames.length - 1, Math.round(time * data.sample_fps)));
      if (frameIndex === lastFrameIndex) return;
      lastFrameIndex = frameIndex;
      const frame = frames[frameIndex];
      const isClose = Math.abs(frame.timestamp - time) < 0.18;
      const visibleHands = isClose ? frame.hands.slice(0, 2) : [];
      const separationShifts = [0, 0];
      if (visibleHands.length === 2) {
        const firstX = (imageCenterX(visibleHands[0]) / VIDEO_WIDTH - 0.5) * 4.4;
        const secondX = (imageCenterX(visibleHands[1]) / VIDEO_WIDTH - 0.5) * 4.4;
        const separation = Math.abs(secondX - firstX);
        if (separation < MIN_HAND_SEPARATION) {
          const extraPerHand = (MIN_HAND_SEPARATION - separation) / 2;
          if (firstX <= secondX) {
            separationShifts[0] = -extraPerHand;
            separationShifts[1] = extraPerHand;
          } else {
            separationShifts[0] = extraPerHand;
            separationShifts[1] = -extraPerHand;
          }
        }
      }
      visuals.forEach((visual, index) => {
        const hand = visibleHands[index];
        if (hand) placeHand(visual, hand, separationShifts[index]);
        else {
          visual.mesh.visible = false;
          visual.joints.visible = false;
          visual.bones.visible = false;
        }
      });
      setCurrentTime(time);
    };

    fetch(`${DATA_BASE}/wilor-hands-60s.json.gzraw?v=compressed-2`)
      .then(async (response) => {
        if (!response.ok) throw new Error("WiLoR data unavailable");
        if (!("DecompressionStream" in window)) {
          const fallback = await fetch(`${DATA_BASE}/wilor-hands-60s.json`);
          if (!fallback.ok) throw new Error("WiLoR data unavailable");
          return fallback.json() as Promise<WiLoRData>;
        }
        const decompressed = response.body?.pipeThrough(new DecompressionStream("gzip"));
        if (!decompressed) throw new Error("WiLoR data was empty");
        return JSON.parse(await new Response(decompressed).text()) as WiLoRData;
      })
      .then((result) => {
        if (stopped) return;
        data = result;
        const grouped = new Map<number, HandMeshRecord[]>();
        result.hands.forEach((hand) => {
          const index = Math.round(hand.timestamp_seconds * result.sample_fps);
          grouped.set(index, [...(grouped.get(index) ?? []), hand]);
        });
        const maxIndex = Math.max(...grouped.keys());
        frames = Array.from({ length: maxIndex + 1 }, (_, index) => ({
          timestamp: index / result.sample_fps,
          hands: grouped.get(index) ?? [],
        }));
        const faceIndex = result.mesh.faces.flat();
        visuals.forEach((visual) => visual.mesh.geometry.setIndex(faceIndex));
        lastFrameIndex = -1;
        setMeshStatus("ready");
      })
      .catch((error: unknown) => {
        if (!stopped) {
          setMeshError(error instanceof Error ? error.message : "Unknown mesh loading error");
          setMeshStatus("error");
        }
      });

    const animate = () => {
      if (stopped) return;
      updateFrame();
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      stopped = true;
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      resetRef.current = null;
      visuals.forEach((visual) => {
        visual.mesh.geometry.dispose();
        visual.mesh.material.dispose();
        visual.joints.geometry.dispose();
        visual.joints.material.dispose();
        visual.bones.geometry.dispose();
        visual.bones.material.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className={`hands3d-shell ${showVideo ? "with-video" : "only-3d"}`}>
      <div className="hands3d-video-panel" aria-hidden={!showVideo}>
        <video
          ref={videoRef}
          src={`${ASSET_BASE}/wilor-overlay-60s.mp4?v=5`}
          poster={`${ASSET_BASE}/wilor-mesh-preview-60s.jpg`}
          controls
          preload="metadata"
          playsInline
          aria-label="Video synchronized to interactive 3D hand reconstruction"
          onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        />
        <span className="hands3d-panel-label">Video + 2D WiLoR overlay</span>
      </div>
      <div className="hands3d-space">
        <div ref={hostRef} className="hands3d-canvas" />
        {meshStatus !== "ready" && (
          <div className={`hands3d-status ${meshStatus}`} role="status">
            {meshStatus === "loading" ? "Loading 3D hand meshes…" : `3D hand meshes could not be loaded${meshError ? `: ${meshError}` : ""}`}
          </div>
        )}
        <div className="hands3d-legend"><span className="left-hand" /> Left <span className="right-hand" /> Right</div>
        <div className="hands3d-time">{formatTime(currentTime)} / 1:00</div>
        <button className="hands3d-reset" onClick={() => resetRef.current?.()}><RotateCcw size={13} /> Reset 3D view</button>
        <span className="hands3d-hint">Drag to orbit · scroll to zoom</span>
      </div>
      <button className="hands3d-video-toggle" onClick={() => setShowVideo((value) => !value)}>
        {showVideo ? <EyeOff size={14} /> : <Eye size={14} />}
        {showVideo ? "Hide video" : "Show video"}
      </button>
    </div>
  );
}
