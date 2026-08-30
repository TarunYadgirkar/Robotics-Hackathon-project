import ARKit
import SceneKit
import SwiftUI

/// An orbitable view of the reconstruction, away from the camera feed.
///
/// The Scan tab shows the mesh pinned to the room through the camera, which is
/// the right view while sweeping. This is the other question -- "what have I
/// actually captured?" -- and it needs the mesh detached from where you are
/// standing.
struct ScanPreview: UIViewRepresentable {
    let scanner: LidarScanner
    let revision: Int
    /// Fallback when this launch has scanned nothing: the cloud already on the
    /// laptop. Without it a relaunched app looks like the scan was lost.
    var serverPoints: [SIMD3<Float>] = []

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = SCNScene()
        view.backgroundColor = .white
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.antialiasingMode = .multisampling2X

        let camera = SCNCamera()
        camera.zNear = 0.01
        camera.zFar = 200
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        view.scene?.rootNode.addChildNode(cameraNode)
        context.coordinator.cameraNode = cameraNode
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        guard context.coordinator.revision != revision else { return }
        context.coordinator.revision = revision
        rebuild(in: view, coordinator: context.coordinator)
    }

    private func rebuild(in view: SCNView, coordinator: Coordinator) {
        guard let scene = view.scene else { return }
        coordinator.meshRoot?.removeFromParentNode()

        let root = SCNNode()
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = UIColor(red: 0.91, green: 0.65, blue: 0.72, alpha: 1)
        material.metalness.contents = 0.0
        material.roughness.contents = 0.55
        material.isDoubleSided = true

        let live = scanner.sceneNodes()
        if !live.isEmpty {
            for node in live {
                node.geometry?.materials = [material]
                root.addChildNode(node)
            }
        } else if !serverPoints.isEmpty {
            root.addChildNode(Self.pointCloudNode(serverPoints))
        }
        scene.rootNode.addChildNode(root)
        coordinator.meshRoot = root

        // Frame the mesh once, rather than leaving the camera at the origin
        // inside the geometry where it looks like nothing was captured.
        if !coordinator.hasFramed, let bounds = bounds(), let camera = coordinator.cameraNode {
            let distance = bounds.radius * 2.6 + 0.8
            camera.simdPosition = bounds.centre + SIMD3<Float>(distance * 0.6, distance * 0.45, distance * 0.7)
            camera.simdLook(
                at: bounds.centre,
                up: SIMD3<Float>(0, 1, 0),
                localFront: SIMD3<Float>(0, 0, -1)
            )
            coordinator.hasFramed = true
        }
    }

    private func bounds() -> (centre: SIMD3<Float>, radius: Float)? {
        if let live = scanner.bounds() { return live }
        guard !serverPoints.isEmpty else { return nil }
        var low = serverPoints[0], high = serverPoints[0]
        for point in serverPoints { low = simd_min(low, point); high = simd_max(high, point) }
        return ((low + high) / 2, max(simd_length(high - low) / 2, 0.5))
    }

    /// SceneKit point cloud. SIMD3<Float> is 16-byte aligned, so the stride is
    /// 16 while only 12 bytes are components -- getting that wrong shears the cloud.
    static func pointCloudNode(_ points: [SIMD3<Float>]) -> SCNNode {
        let stride = MemoryLayout<SIMD3<Float>>.stride
        let data = points.withUnsafeBytes { Data($0) }
        let source = SCNGeometrySource(
            data: data, semantic: .vertex, vectorCount: points.count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0, dataStride: stride
        )

        let indices = (0..<UInt32(points.count)).map { $0 }
        let element = SCNGeometryElement(
            data: indices.withUnsafeBytes { Data($0) },
            primitiveType: .point, primitiveCount: points.count,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )
        element.pointSize = 4
        element.minimumPointScreenSpaceRadius = 1.5
        element.maximumPointScreenSpaceRadius = 4

        let geometry = SCNGeometry(sources: [source], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(red: 0.76, green: 0.09, blue: 0.24, alpha: 1)
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var meshRoot: SCNNode?
        var cameraNode: SCNNode?
        var revision = -1
        var hasFramed = false
    }
}
