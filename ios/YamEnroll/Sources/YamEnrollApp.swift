import SwiftUI

@main
struct YamEnrollApp: App {
    @StateObject private var client = ServerClient()
    @StateObject private var scanner = LidarScanner()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(client)
                .environmentObject(scanner)
                .preferredColorScheme(.light)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var client: ServerClient

    var body: some View {
        if client.isConnected {
            MainTabs()
        } else {
            ConnectView()
        }
    }
}

struct MainTabs: View {
    @State private var tab = 0

    var body: some View {
        TabView(selection: $tab) {
            EnrollView()
                .tabItem { Label("Enroll", systemImage: "hand.point.up.left") }
                .tag(0)
            ScanView()
                .tabItem { Label("Scan", systemImage: "cube.transparent") }
                .tag(1)
        }
        .tint(Theme.red)
    }
}

enum Theme {
    static let red = Color(red: 0.76, green: 0.09, blue: 0.24)
    static let ink = Color(red: 0.09, green: 0.07, blue: 0.08)
    static let muted = Color(red: 0.43, green: 0.39, blue: 0.41)
    static let hairline = Color(red: 0.90, green: 0.89, blue: 0.89)
}
