import SwiftUI

/// The touch-to-capture half, driven from the arm's live joint feedback.
struct EnrollView: View {
    @EnvironmentObject private var client: ServerClient
    @EnvironmentObject private var scanner: LidarScanner
    @State private var confirmingFinish = false
    @State private var serverPoints: [SIMD3<Float>] = []

    private var state: EnrollmentState? { client.state }

    var body: some View {
        Group {
            if state?.isFinished == true {
                FinishedView(state: state)
            } else {
                enrolling
            }
        }
        .background(Color.white)
    }

    private var enrolling: some View {
        VStack(spacing: 0) {
            header
            preview
            referenceHint
            stats
            controls
        }
        .padding(.horizontal, 20)
        .confirmationDialog("End enrollment?", isPresented: $confirmingFinish, titleVisibility: .visible) {
            Button("Save and finish", role: .destructive) {
                Task { await client.send(command: "done") }
            }
            Button("Keep enrolling", role: .cancel) {}
        } message: {
            Text("This saves the session and releases the arm. It does not upload the scan -- send that from the Scan tab first.")
        }
    }

    /// The reconstruction so far, alongside the coverage it is being judged by.
    private var preview: some View {
        ZStack(alignment: .topTrailing) {
            if scanner.meshCount > 0 || !serverPoints.isEmpty {
                ScanPreview(scanner: scanner,
                            revision: scanner.meshCount + serverPoints.count,
                            serverPoints: serverPoints)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
            } else {
                RoundedRectangle(cornerRadius: 18)
                    .fill(Color(white: 0.97))
                    .overlay(
                        VStack(spacing: 6) {
                            Image(systemName: "cube.transparent")
                                .font(.system(size: 26)).foregroundStyle(Theme.muted)
                            Text("Sweep the room on the Scan tab\nto build the mesh")
                                .multilineTextAlignment(.center)
                                .font(.system(size: 12)).foregroundStyle(Theme.muted)
                        }
                    )
            }
            CoverageDial(patches: state?.patches ?? [], progress: state?.progress ?? 0)
                .frame(width: 108, height: 108)
                .padding(10)
        }
        .frame(maxHeight: .infinity)
        .padding(.vertical, 10)
        .task {
            // A relaunched app has no anchors in memory, but the laptop still
            // holds the scan; without this the Enroll tab looks empty.
            if scanner.meshCount == 0, let points = try? await client.fetchScanPoints() {
                serverPoints = points
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("OBSTACLE ENROLLMENT")
                .font(.system(size: 11, weight: .medium)).tracking(2)
                .foregroundStyle(Theme.muted)
            Text(state?.objectName ?? "—")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(Theme.ink)
            Text(state?.prompt ?? "Waiting for the arm…")
                .font(.system(size: 14))
                .foregroundStyle(Theme.muted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 12)
    }

    private var referenceHint: some View {
        let count = state?.references?.count ?? 0
        let hasScan = !(state?.scans ?? []).isEmpty
        return Group {
            if hasScan && count < 3 {
                HStack(spacing: 8) {
                    Image(systemName: "scope").font(.system(size: 12))
                    Text(count == 0
                         ? "To align the scan: put the tip on a corner you can also see in the scan, tap Reference, then tap that same corner on the Scan tab. 3 needed."
                         : "\(count) of 3 references. Tap the matching spot on the Scan tab, then set the next one.")
                        .font(.system(size: 12))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .foregroundStyle(Theme.red)
                .padding(.horizontal, 12).padding(.vertical, 9)
                .background(Color(red: 0.98, green: 0.93, blue: 0.94), in: RoundedRectangle(cornerRadius: 12))
                .padding(.bottom, 8)
            }
        }
    }

    private var stats: some View {
        HStack(spacing: 18) {
            stat("points", "\(state?.pointCount ?? 0)")
            stat("directions", "\(state?.patches?.filter { $0 }.count ?? 0)")
            stat("poses", "\(state?.poseSamples ?? 0)")
            stat("refs", "\(state?.references?.count ?? 0)/3")
            if let temperature = state?.temperature {
                stat("temp", "\(Int(temperature))°C")
            }
        }
        .padding(.bottom, 10)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 15, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.red)
            Text(label).font(.system(size: 10)).foregroundStyle(Theme.muted)
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            Button {
                Haptics.tap()
                Task { await client.send(command: "capture") }
            } label: {
                Text("Capture point")
                    .font(.system(size: 17, weight: .semibold))
                    .frame(maxWidth: .infinity).padding(.vertical, 17)
                    .background(Theme.red, in: Capsule())
                    .foregroundStyle(.white)
            }

            HStack(spacing: 10) {
                secondary("Reference", "scope") { Task { await client.send(command: "reference") } }
                secondary("Undo", "arrow.uturn.backward") { Task { await client.send(command: "undo") } }
                secondary("Next", "plus") { Task { await client.send(command: "next") } }
                secondary("Finish", "checkmark") { confirmingFinish = true }
            }
        }
        .padding(.bottom, 8)
    }

    private func secondary(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button {
            Haptics.tap()
            action()
        } label: {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 14))
                Text(title).font(.system(size: 11, weight: .medium))
            }
            .frame(maxWidth: .infinity).padding(.vertical, 11)
            .background(Color(white: 0.96), in: RoundedRectangle(cornerRadius: 14))
            .foregroundStyle(Theme.ink)
        }
    }
}

/// The coverage shell, flattened for a phone: one wedge per direction the object
/// has been touched from, lit as it fills.
struct CoverageDial: View {
    let patches: [Bool]
    let progress: Double

    var body: some View {
        GeometryReader { geometry in
            let side = min(geometry.size.width, geometry.size.height)
            let radius = side / 2 - 18
            ZStack {
                ForEach(patches.indices, id: \.self) { index in
                    let lit = patches[index]
                    let band = index / 8            // elevation band
                    let sector = index % 8          // azimuth
                    let inner = radius * (0.42 + 0.19 * Double(band))
                    let outer = inner + radius * 0.17
                    Wedge(startAngle: .degrees(Double(sector) * 45 - 90),
                          endAngle: .degrees(Double(sector) * 45 + 43 - 90),
                          innerRadius: inner, outerRadius: outer)
                        .fill(lit ? Theme.red : Color(white: 0.90))
                        .opacity(lit ? 0.95 : 0.55)
                        .animation(.easeOut(duration: 0.35), value: lit)
                }
                VStack(spacing: 1) {
                    Text("\(Int(progress * 100))%")
                        .font(.system(size: 26, weight: .semibold, design: .rounded))
                        .foregroundStyle(Theme.ink)
                    Text("coverage").font(.system(size: 10)).foregroundStyle(Theme.muted)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}

struct Wedge: Shape {
    let startAngle: Angle
    let endAngle: Angle
    let innerRadius: Double
    let outerRadius: Double

    func path(in rect: CGRect) -> Path {
        let centre = CGPoint(x: rect.midX, y: rect.midY)
        var path = Path()
        path.addArc(center: centre, radius: outerRadius, startAngle: startAngle, endAngle: endAngle, clockwise: false)
        path.addArc(center: centre, radius: innerRadius, startAngle: endAngle, endAngle: startAngle, clockwise: true)
        path.closeSubpath()
        return path
    }
}

enum Haptics {
    static func tap() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
    static func success() {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}


/// What Finish actually did. Previously the server stopped on Finish and the
/// phone simply went quiet, which reads as a failed upload rather than a save.
struct FinishedView: View {
    let state: EnrollmentState?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 44)).foregroundStyle(Theme.red)
            Text("Enrollment saved")
                .font(.system(size: 28, weight: .semibold)).foregroundStyle(Theme.ink)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(state?.objects ?? [], id: \.name) { object in
                    row(object.name, object.boxMm.map { "\($0[0]) x \($0[1]) x \($0[2]) mm" }
                        ?? "\(object.points) points")
                }
                row("arm poses logged", "\(state?.poseSamples ?? 0)")
                row("scans uploaded", "\((state?.scans ?? []).count)")
                row("reference points", "\(state?.referenceCount ?? 0)")
            }

            if (state?.referenceCount ?? 0) < 3, !(state?.scans ?? []).isEmpty {
                Text("The scan needs 3 reference points to be aligned to the arm. Without them it cannot be used for planning.")
                    .font(.system(size: 13)).foregroundStyle(Theme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("The arm has been released. Restart the server on the laptop to enroll again.")
                .font(.system(size: 13)).foregroundStyle(Theme.muted)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.system(size: 14)).foregroundStyle(Theme.muted)
            Spacer()
            Text(value).font(.system(size: 14, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.ink)
        }
    }
}
