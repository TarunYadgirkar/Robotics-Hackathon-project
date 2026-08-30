import SwiftUI

struct ConnectView: View {
    @EnvironmentObject private var client: ServerClient
    @State private var text = ""
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Spacer()

            Text("YAM ENROLL")
                .font(.system(size: 12, weight: .medium)).tracking(2)
                .foregroundStyle(Theme.muted)
            Text("Connect to the arm")
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(Theme.ink)
            Text("Paste the link the laptop printed. It already contains the access token.")
                .font(.system(size: 15))
                .foregroundStyle(Theme.muted)

            TextField("https://….trycloudflare.com/?k=…", text: $text, axis: .vertical)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .font(.system(size: 15, design: .monospaced))
                .padding(14)
                .background(Color(white: 0.97), in: RoundedRectangle(cornerRadius: 12))

            Button {
                busy = true
                Task { await client.connect(to: text); busy = false }
            } label: {
                HStack {
                    if busy { ProgressView().tint(.white) }
                    Text(busy ? "Connecting…" : "Connect")
                }
                .font(.system(size: 16, weight: .semibold))
                .frame(maxWidth: .infinity).padding(.vertical, 15)
                .background(Theme.red, in: Capsule())
                .foregroundStyle(.white)
            }
            .disabled(text.isEmpty || busy)
            .opacity(text.isEmpty ? 0.4 : 1)

            if let error = client.lastError {
                Text(error).font(.system(size: 13)).foregroundStyle(Theme.red)
            }

            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white)
    }
}
