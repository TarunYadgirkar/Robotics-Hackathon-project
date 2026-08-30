import Foundation

/// Talks to the enrollment server running next to the arm.
///
/// The laptop holds the CAN link, so every joint angle and every captured point
/// comes from there; this app is the operator's hands and the room's scanner.
@MainActor
final class ServerClient: ObservableObject {
    @Published var baseURL: URL?
    @Published var token: String = ""
    @Published var isConnected = false
    @Published var lastError: String?
    @Published var state: EnrollmentState?

    private var poller: Task<Void, Never>?
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 8
        configuration.waitsForConnectivity = false
        return URLSession(configuration: configuration)
    }()

    /// Accepts the whole link the server prints, token and all.
    func connect(to text: String) async {
        lastError = nil
        guard var components = URLComponents(string: normalized(text)) else {
            lastError = "That does not look like a URL."
            return
        }

        if let queryToken = components.queryItems?.first(where: { $0.name == "k" })?.value {
            token = queryToken
        }
        components.query = nil
        components.path = ""

        guard let url = components.url else {
            lastError = "That does not look like a URL."
            return
        }

        baseURL = url
        do {
            state = try await fetchState()
            isConnected = true
            startPolling()
        } catch let error as ClientError {
            isConnected = false
            lastError = error.errorDescription
        } catch {
            isConnected = false
            lastError = "Could not reach \(url.host ?? "the server"). \(error.localizedDescription)"
        }
    }

    func disconnect() {
        poller?.cancel()
        poller = nil
        isConnected = false
        state = nil
    }

    private func normalized(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") { return trimmed }
        return "http://" + trimmed
    }

    private func startPolling() {
        poller?.cancel()
        poller = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                if let fresh = try? await self.fetchState() {
                    self.state = fresh
                    self.lastError = nil
                } else {
                    self.lastError = "Lost contact with the server."
                }
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
    }

    private func request(_ path: String, method: String = "GET") -> URLRequest? {
        guard let baseURL, let url = URL(string: path, relativeTo: baseURL) else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if !token.isEmpty { request.setValue(token, forHTTPHeaderField: "X-Token") }
        return request
    }

    func fetchState() async throws -> EnrollmentState {
        guard let request = request("/api/state") else { throw ClientError.notConfigured }
        let (data, response) = try await session.data(for: request)
        try Self.check(response, data: data)
        do {
            return try JSONDecoder().decode(EnrollmentState.self, from: data)
        } catch {
            throw ClientError.notTheServer
        }
    }

    /// Turn transport-level failures into something the operator can act on.
    ///
    /// A dead tunnel answers with a Cloudflare error *page*, so the raw failure
    /// surfaces as a JSON decoding error -- "the data couldn't be read because
    /// it isn't in the correct format" -- which says nothing about the actual
    /// problem, that the link is stale.
    static func check(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        switch http.statusCode {
        case 200..<300:
            return
        case 401:
            throw ClientError.badToken
        case 502, 503, 530:
            throw ClientError.tunnelDown
        default:
            throw ClientError.httpStatus(http.statusCode)
        }
    }

    enum ClientErrorKind { case configuration, token, tunnel, format }

    @discardableResult
    func send(command: String) async -> Bool {
        guard var request = request("/api/command", method: "POST") else { return false }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["action": command])
        return (try? await session.data(for: request)) != nil
    }

    func upload(scan data: Data, named name: String) async throws -> ScanSummary {
        guard var request = request("/api/scan", method: "POST") else { throw ClientError.notConfigured }
        request.setValue(name, forHTTPHeaderField: "X-Filename")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        let (body, response) = try await session.upload(for: request, from: data)
        try Self.check(response, data: body)
        return try JSONDecoder().decode(ScanSummary.self, from: body)
    }

    func register(scanPoints: [[Double]], robotPoints: [[Double]]) async throws -> Registration {
        guard var request = request("/api/register", method: "POST") else { throw ClientError.notConfigured }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: ["scan": scanPoints, "robot": robotPoints]
        )
        let (data, response) = try await session.data(for: request)
        try Self.check(response, data: data)
        return try JSONDecoder().decode(Registration.self, from: data)
    }

    enum ClientError: LocalizedError {
        case notConfigured
        case badToken
        case tunnelDown
        case notTheServer
        case httpStatus(Int)

        var errorDescription: String? {
            switch self {
            case .notConfigured:
                return "Not connected to a server yet."
            case .badToken:
                return "The server rejected that token. Copy the whole link again -- a new one is issued every time the server restarts."
            case .tunnelDown:
                return "That tunnel is no longer running. Restarting the server on the laptop mints a new link; copy the latest one."
            case .notTheServer:
                return "Reached something that is not the enrollment server. Check the link is the one the laptop printed."
            case .httpStatus(let code):
                return "The server answered with HTTP \(code)."
            }
        }
    }
}

struct EnrollmentState: Decodable {
    var status: String?
    var objectName: String?
    var prompt: String?
    var joints: [Double]?
    var tip: [Double]?
    var points: [[Double]]?
    var references: [[Double]]?
    var pointCount: Int?
    var patches: [Bool]?
    var progress: Double?
    var temperature: Double?
    var poseSamples: Int?
    var scans: [String]?
    var message: String?

    enum CodingKeys: String, CodingKey {
        case status, prompt, joints, tip, points, references, patches, progress, temperature, scans, message
        case objectName = "object_name"
        case pointCount = "point_count"
        case poseSamples = "pose_samples"
    }
}

struct ScanSummary: Decodable {
    var points: Int?
    var bytes: Int?
    var parseError: String?
    enum CodingKeys: String, CodingKey { case points, bytes, parseError = "parse_error" }
}

struct Registration: Decodable {
    var rmseMm: Double?
    var pairs: Int?
    var trustworthy: Bool?
    var error: String?
    enum CodingKeys: String, CodingKey { case pairs, trustworthy, error, rmseMm = "rmse_mm" }
}
