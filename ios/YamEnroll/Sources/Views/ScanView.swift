import ARKit
import RealityKit
import SwiftUI

/// The LiDAR half: sweep the workcell, then tie the scan to the robot's frame.
///
/// Registration happens here rather than on the laptop because this is where
/// the operator can see the feature they are pointing at. Touch a landmark with
/// the arm (Reference, on the Enroll tab), then tap that same landmark on this
/// screen; three pairs fix the transform.
struct ScanView: View {
    @EnvironmentObject private var scanner: LidarScanner
    @EnvironmentObject private var client: ServerClient

    @State private var pairs: [(scan: [Double], robot: [Double])] = []
    @State private var status: String?
    @State private var statusIsWarning = false
    @State private var uploading = false
    @State private var uploadedPoints: Int?
    @State private var erasing = false

    private var references: [[Double]] { client.state?.references ?? [] }

    var body: some View {
        ZStack(alignment: .top) {
            ARMeshView(scanner: scanner, onTap: handleTap)
                .ignoresSafeArea()

            VStack(spacing: 10) {
                banner
                Spacer()
                if let status {
                    Text(status)
                        .font(.system(size: 13, weight: .medium))
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(statusIsWarning ? Theme.red : Theme.ink, in: Capsule())
                        .foregroundStyle(.white)
                }
                controls
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .onAppear { if !scanner.isScanning { scanner.start() } }
        .onDisappear { scanner.pause() }
    }

    private var banner: some View {
        VStack(spacing: 6) {
            HStack(spacing: 14) {
                metric("meshes", "\(scanner.meshCount)")
                metric("vertices", format(scanner.vertexCount))
                metric("paired", "\(pairs.count)/3")
                metric("refs", "\(references.count)")
                if let uploadedPoints { metric("uploaded", format(uploadedPoints)) }
            }
            if let note = scanner.trackingNote {
                Text(note).font(.system(size: 12, weight: .medium)).foregroundStyle(.white)
            } else if !scanner.isSupported {
                Text("No LiDAR on this device").font(.system(size: 12)).foregroundStyle(.white)
            } else if erasing {
                Text("Erase mode — tap to delete a 25cm ball of scan")
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.red)
            } else if uploadedPoints != nil && pairs.count < 3 {
                Text(references.isEmpty
                     ? "Set a Reference on the Enroll tab first"
                     : "Tap the spot matching reference \(pairs.count + 1)")
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(.white)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.top, 6)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 1) {
            Text(value).font(.system(size: 14, weight: .semibold, design: .monospaced))
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary)
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Button {
                Haptics.tap()
                scanner.reset()
                pairs.removeAll()
                show("scan cleared")
            } label: { label("Clear", "trash") }

            Button {
                Haptics.tap()
                Task { await upload() }
            } label: {
                HStack(spacing: 6) {
                    if uploading { ProgressView().tint(.white) }
                    Text(uploading ? "Uploading…" : "Send scan")
                }
                .font(.system(size: 16, weight: .semibold))
                .frame(maxWidth: .infinity).padding(.vertical, 15)
                .background(Theme.red, in: Capsule())
                .foregroundStyle(.white)
            }
            .disabled(uploading || scanner.vertexCount == 0)
            .opacity(scanner.vertexCount == 0 ? 0.45 : 1)

            Button {
                Haptics.tap()
                erasing.toggle()
                show(erasing
                     ? "tap anything that should not be in the map"
                     : "back to pairing points")
            } label: {
                label(erasing ? "Erasing" : "Erase", "eraser")
                    .overlay(RoundedRectangle(cornerRadius: 14)
                        .stroke(erasing ? Theme.red : .clear, lineWidth: 2))
            }
        }
    }

    private func label(_ title: String, _ icon: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 14))
            Text(title).font(.system(size: 10, weight: .medium))
        }
        .frame(width: 66).padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .foregroundStyle(Theme.ink)
    }

    private func handleTap(_ worldPoint: SIMD3<Float>) {
        if erasing {
            Task {
                do {
                    let removed = try await client.eraseScan(centre: worldPoint, radius: 0.25)
                    Haptics.tap()
                    show(removed > 0 ? "erased \(removed) points" : "nothing there to erase")
                } catch {
                    show("erase failed: \(error.localizedDescription)", warning: true)
                }
            }
            return
        }
        guard uploadedPoints != nil else {
            show("send the scan first, then pair points", warning: true); return
        }
        guard pairs.count < references.count else {
            show("touch another Reference point with the arm first", warning: true); return
        }
        pairs.append((
            scan: [Double(worldPoint.x), Double(worldPoint.y), Double(worldPoint.z)],
            robot: references[pairs.count]
        ))
        Haptics.tap()

        if pairs.count >= 3 {
            Task { await register() }
        } else {
            show("paired \(pairs.count) of 3")
        }
    }

    private func upload() async {
        uploading = true
        defer { uploading = false }

        let points = scanner.worldPoints()
        guard !points.isEmpty else { show("nothing scanned yet", warning: true); return }

        do {
            let summary = try await client.upload(
                scan: PLYWriter.data(from: points),
                named: "phone_scan_\(Int(Date().timeIntervalSince1970)).ply"
            )
            uploadedPoints = summary.points ?? points.count
            Haptics.success()
            show("sent \(format(summary.points ?? points.count)) points — now pair 3 landmarks")
        } catch {
            show("upload failed: \(error.localizedDescription)", warning: true)
        }
    }

    private func register() async {
        do {
            let result = try await client.register(
                scanPoints: pairs.map(\.scan),
                robotPoints: pairs.map(\.robot)
            )
            if let error = result.error {
                show(error, warning: true)
                pairs.removeAll()
                return
            }
            let rmse = result.rmseMm ?? 0
            if result.trustworthy == true {
                Haptics.success()
                show(String(format: "aligned to %.1f mm", rmse))
            } else {
                show(String(format: "aligned but %.0f mm off — re-pair", rmse), warning: true)
                pairs.removeAll()
            }
        } catch {
            show("alignment failed: \(error.localizedDescription)", warning: true)
            pairs.removeAll()
        }
    }

    private func show(_ text: String, warning: Bool = false) {
        status = text
        statusIsWarning = warning
        Task {
            try? await Task.sleep(for: .seconds(3))
            if status == text { status = nil }
        }
    }

    private func format(_ value: Int) -> String {
        value >= 1000 ? String(format: "%.1fk", Double(value) / 1000) : "\(value)"
    }
}

/// RealityKit host showing the live reconstruction over the camera feed.
///
/// RealityKit rather than SceneKit because `.showSceneUnderstanding` renders the
/// reconstructed mesh for free; SceneKit has no equivalent and would mean
/// building SCNGeometry from every anchor by hand just to draw a wireframe.
struct ARMeshView: UIViewRepresentable {
    let scanner: LidarScanner
    let onTap: (SIMD3<Float>) -> Void

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.session = scanner.session
        view.debugOptions = [.showSceneUnderstanding]
        view.environment.sceneUnderstanding.options = [.occlusion]
        context.coordinator.view = view

        let recognizer = UITapGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handleTap(_:))
        )
        view.addGestureRecognizer(recognizer)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.onTap = onTap
    }

    func makeCoordinator() -> Coordinator { Coordinator(scanner: scanner, onTap: onTap) }

    final class Coordinator: NSObject {
        weak var view: ARView?
        let scanner: LidarScanner
        var onTap: (SIMD3<Float>) -> Void

        init(scanner: LidarScanner, onTap: @escaping (SIMD3<Float>) -> Void) {
            self.scanner = scanner
            self.onTap = onTap
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let view else { return }
            let location = recognizer.location(in: view)
            Task { @MainActor in
                if let point = scanner.hitTest(location, in: view) { onTap(point) }
            }
        }
    }
}
