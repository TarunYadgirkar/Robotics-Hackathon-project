import type { Metadata } from "next";
import { Explorer } from "./components/Explorer";

export const metadata: Metadata = {
  title: "Factory Atlas — One-video reconstruction proof",
  description:
    "Explore precomputed factory video, hand-joint, and relative-depth 3D data.",
};

export default function Home() {
  return <Explorer />;
}
