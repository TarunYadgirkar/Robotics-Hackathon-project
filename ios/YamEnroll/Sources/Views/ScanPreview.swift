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

        for node in scanner.sceneNodes() {
            node.geometry?.materials = [material]
            root.addChildNode(node)
        }
        scene.rootNode.addChildNode(root)
        coordinator.meshRoot = root

        // Frame the mesh once, rather than leaving the camera at the origin
        // inside the geometry where it looks like nothing was captured.
        if !coordinator.hasFramed, let bounds = scanner.bounds(), let camera = coordinator.cameraNode {
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

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var meshRoot: SCNNode?
        var cameraNode: SCNNode?
        var revision = -1
        var hasFramed = false
    }
}
