import SwiftUI

/// The touch-to-capture half, driven from the arm's live joint feedback.
struct EnrollView: View {
    @EnvironmentObject private var client: ServerClient

    private var state: EnrollmentState? { client.state }

    var body: some View {
        VStack(spacing: 0) {
            header
            Spacer(minLength: 0)
            CoverageDial(patches: state?.patches ?? [], progress: state?.progress ?? 0)
                .frame(height: 220)
            Spacer(minLength: 0)
            stats
            controls
        }
        .padding(.horizontal, 20)
        .background(Color.white)
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

    private var stats: some View {
        HStack(spacing: 18) {
            stat("points", "\(state?.pointCount ?? 0)")
            stat("directions", "\(state?.patches?.filter { $0 }.count ?? 0)")
            stat("poses", "\(state?.poseSamples ?? 0)")
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
                secondary("Reference", "scope") { await client.send(command: "reference") }
                secondary("Undo", "arrow.uturn.backward") { await client.send(command: "undo") }
                secondary("Next", "plus") { await client.send(command: "next") }
                secondary("Finish", "checkmark") { await client.send(command: "done") }
            }
        }
        .padding(.bottom, 8)
    }

    private func secondary(_ title: String, _ icon: String, action: @escaping () async -> Void) -> some View {
        Button {
            Haptics.tap()
            Task { await action() }
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
