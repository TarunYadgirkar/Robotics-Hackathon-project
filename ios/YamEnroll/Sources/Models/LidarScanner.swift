import ARKit
import Combine
import RealityKit
import simd

/// Captures the workcell with the phone's LiDAR.
///
/// ARKit's scene reconstruction hands back a set of `ARMeshAnchor`s that it
/// keeps revising as you walk around, so anchors are stored by identifier and
/// overwritten rather than appended -- appending would accumulate several
/// versions of the same wall.
@MainActor
final class LidarScanner: NSObject, ObservableObject {
    @Published private(set) var isScanning = false
    @Published private(set) var isSupported = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
    @Published private(set) var meshCount = 0
    @Published private(set) var vertexCount = 0
    @Published private(set) var trackingNote: String?

    /// Latest geometry per anchor, in ARKit world coordinates.
    private var anchors: [UUID: ARMeshAnchor] = [:]
    let session = ARSession()

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard isSupported else {
            trackingNote = "This device has no LiDAR scanner."
            return
        }
        let configuration = ARWorldTrackingConfiguration()
        configuration.sceneReconstruction = .mesh
        configuration.environmentTexturing = .none
        // The scan is geometry, not photography; depth frames are what matter.
        configuration.frameSemantics = []
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        anchors.removeAll()
        meshCount = 0
        vertexCount = 0
        isScanning = true
    }

    func pause() {
        session.pause()
        isScanning = false
    }

    func reset() {
        anchors.removeAll()
        meshCount = 0
        vertexCount = 0
        if isScanning { start() }
    }

    /// Every mesh vertex in world space, thinned onto a voxel grid.
    ///
    /// Raw scene reconstruction runs to hundreds of thousands of vertices with
    /// heavy overlap between anchors. Collapsing them onto a grid at roughly the
    /// planner's own resolution keeps the upload small without losing anything
    /// the planner could act on.
    func worldPoints(voxelSize: Float = 0.01) -> [SIMD3<Float>] {
        var grid: [SIMD3<Int32>: SIMD3<Float>] = [:]
        grid.reserveCapacity(64_000)

        for anchor in anchors.values {
            let geometry = anchor.geometry
            let vertices = geometry.vertices
            let buffer = vertices.buffer.contents()
            let transform = anchor.transform

            for index in 0..<vertices.count {
                let pointer = buffer.advanced(by: vertices.offset + vertices.stride * index)
                let local = pointer.assumingMemoryBound(to: SIMD3<Float>.self).pointee
                let world = transform * SIMD4<Float>(local, 1)
                let position = SIMD3<Float>(world.x, world.y, world.z)
                let key = SIMD3<Int32>(
                    Int32((position.x / voxelSize).rounded()),
                    Int32((position.y / voxelSize).rounded()),
                    Int32((position.z / voxelSize).rounded())
                )
                grid[key] = position
            }
        }
        return Array(grid.values)
    }

    /// Where a screen tap lands on the scanned surface, in ARKit world space.
    func hitTest(_ point: CGPoint, in view: ARView) -> SIMD3<Float>? {
        guard let query = view.makeRaycastQuery(from: point, allowing: .estimatedPlane, alignment: .any),
              let result = session.raycast(query).first else { return nil }
        let t = result.worldTransform
        return SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
    }
}

extension LidarScanner: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        store(anchors)
    }

    nonisolated func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        store(anchors)
    }

    nonisolated func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        let ids = anchors.compactMap { ($0 as? ARMeshAnchor)?.identifier }
        Task { @MainActor in
            for id in ids { self.anchors.removeValue(forKey: id) }
            self.recount()
        }
    }

    nonisolated func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let note: String?
        switch camera.trackingState {
        case .normal: note = nil
        case .notAvailable: note = "Tracking unavailable"
        case .limited(.excessiveMotion): note = "Slow down -- moving too fast to map"
        case .limited(.insufficientFeatures): note = "Not enough detail here to track"
        case .limited(.initializing): note = "Starting up, move the phone gently"
        case .limited: note = "Tracking limited"
        }
        Task { @MainActor in self.trackingNote = note }
    }

    private nonisolated func store(_ incoming: [ARAnchor]) {
        let meshes = incoming.compactMap { $0 as? ARMeshAnchor }
        guard !meshes.isEmpty else { return }
        Task { @MainActor in
            for mesh in meshes { self.anchors[mesh.identifier] = mesh }
            self.recount()
        }
    }

    @MainActor
    private func recount() {
        meshCount = anchors.count
        vertexCount = anchors.values.reduce(0) { $0 + $1.geometry.vertices.count }
    }
}
