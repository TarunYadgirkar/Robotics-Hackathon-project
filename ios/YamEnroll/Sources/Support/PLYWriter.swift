import Foundation
import simd

/// Serialises a point cloud as binary little-endian PLY.
///
/// Binary rather than ASCII purely for size: a 60k-point cloud is about 700KB
/// binary against 2MB of text, over a phone connection that may be a tunnel.
/// The server's loader already reads both.
enum PLYWriter {
    static func data(from points: [SIMD3<Float>]) -> Data {
        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "comment written by YAM Enroll\n"
        header += "element vertex \(points.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "end_header\n"

        var output = Data(header.utf8)
        output.reserveCapacity(output.count + points.count * 12)
        for point in points {
            var x = point.x, y = point.y, z = point.z
            withUnsafeBytes(of: &x) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &y) { output.append(contentsOf: $0) }
            withUnsafeBytes(of: &z) { output.append(contentsOf: $0) }
        }
        return output
    }
}
