import SwiftUI
import AVFoundation
import ReplayKit

struct ContentView: View {
    @StateObject private var camera = CameraService()
    @State private var isRecording = false

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            GeometryReader { _ in
                Canvas { ctx, size in
                    for det in camera.detections.prefix(15) {
                        let r = det.rect
                        let box = CGRect(
                            x: r.minX * size.width,
                            y: r.minY * size.height,
                            width: r.width * size.width,
                            height: r.height * size.height
                        )
                        let color = boxColor(for: det.classIndex)
                        ctx.stroke(Path(box), with: .color(color), lineWidth: 2)
                        var text = AttributedString(det.label + " " + String(Int(det.confidence * 100)) + "%")
                        text.font = .systemFont(ofSize: 11, weight: .semibold)
                        text.foregroundColor = .white
                        let resolved = ctx.resolve(Text(text))
                        let sz = resolved.measure(in: CGSize(width: 200, height: 20))
                        let lr = CGRect(x: box.minX, y: max(0, box.minY - 18),
                                        width: sz.width + 6, height: 18)
                        ctx.fill(Path(lr), with: .color(color))
                        ctx.draw(resolved, at: CGPoint(x: lr.minX + 3, y: lr.minY + 2))
                    }
                }
            }
            .ignoresSafeArea()

            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(String(format: "%.0f FPS", camera.fps))
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .foregroundStyle(.green)
                        Text(String(format: "%.1f ms", camera.inferenceMs))
                            .font(.system(size: 12, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.8))
                        Text(String(camera.detections.count) + " objects")
                            .font(.system(size: 12, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .padding(10)
                    .background(.black.opacity(0.5))
                    .cornerRadius(10)
                    .padding()
                }
                Spacer()
            }

            if !isRecording {
                VStack {
                    Spacer()
                    Button(action: toggleRecording) {
                        Image(systemName: "record.circle")
                            .font(.system(size: 56))
                            .foregroundStyle(.white)
                            .shadow(radius: 4)
                    }
                    .padding(.bottom, 40)
                }
            }

            if camera.permissionDenied {
                VStack(spacing: 12) {
                    Image(systemName: "camera.slash.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Camera access required")
                        .font(.headline)
                    Text("Enable in Settings → Privacy → Camera")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                .background(.ultraThinMaterial)
                .cornerRadius(16)
            }
        }
        .onTapGesture { if isRecording { toggleRecording() } }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    private func toggleRecording() {
        let recorder = RPScreenRecorder.shared()
        if isRecording {
            recorder.stopRecording { preview, _ in
                isRecording = false
                guard let preview else { return }
                preview.previewControllerDelegate = RecordingDelegate.shared
                DispatchQueue.main.async {
                    UIApplication.shared.connectedScenes
                        .compactMap { $0 as? UIWindowScene }
                        .first?.windows.first?.rootViewController?
                        .present(preview, animated: true)
                }
            }
        } else {
            recorder.startRecording { _ in
                DispatchQueue.main.async { isRecording = true }
            }
        }
    }

    private func boxColor(for classIndex: Int) -> Color {
        let colors: [Color] = [.blue, .green, .orange, .pink, .purple, .red, .teal, .yellow]
        return colors[classIndex % colors.count]
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

final class RecordingDelegate: NSObject, RPPreviewViewControllerDelegate {
    static let shared = RecordingDelegate()
    func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
        previewController.dismiss(animated: true)
    }
}
