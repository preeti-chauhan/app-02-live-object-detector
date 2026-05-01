import AVFoundation
import CoreML
import UIKit
import Combine

struct Detection {
    let classIndex: Int
    let confidence: Float
    let rect: CGRect  // normalized [0,1] x, y, width, height (top-left origin)

    var label: String { CameraService.classes[classIndex] }
}

final class CameraService: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    // MARK: - Published state
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var inferenceMs: Double = 0
    @Published var permissionDenied = false

    // MARK: - Camera + model
    let session = AVCaptureSession()
    private let model: yolov8n
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "camera.processing", qos: .userInteractive)

    // FPS tracking
    private var frameCount = 0
    private var lastFPSTime = CACurrentMediaTime()

    // Skip frames to avoid backlog — run inference every other frame
    private var frameIndex = 0

    // COCO 80 classes
    static let classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    override init() {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif
        guard let m = try? yolov8n(configuration: config) else {
            fatalError("Failed to load yolov8n.mlpackage")
        }
        self.model = m
        super.init()
    }

    // MARK: - Session setup

    func start() {
        checkPermission()
    }

    private func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted { self?.setupSession() }
                else { DispatchQueue.main.async { self?.permissionDenied = true } }
            }
        default:
            DispatchQueue.main.async { self.permissionDenied = true }
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                           ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                        kCVPixelFormatType_32BGRA]
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90  // portrait
        }

        session.commitConfiguration()
        processingQueue.async { self.session.startRunning() }
    }

    func stop() {
        processingQueue.async { self.session.stopRunning() }
    }

    // MARK: - Frame processing

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // FPS counter — every frame
        frameCount += 1
        let now = CACurrentMediaTime()
        let elapsed = now - lastFPSTime
        if elapsed >= 1.0 {
            let currentFPS = Double(frameCount) / elapsed
            frameCount = 0
            lastFPSTime = now
            DispatchQueue.main.async { self.fps = currentFPS }
        }

        // Run inference every 2nd frame to prevent backlog
        frameIndex += 1
        guard frameIndex % 2 == 0 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let resized = resize(pixelBuffer, to: CGSize(width: 640, height: 640)) else { return }

        let t0 = CACurrentMediaTime()
        guard let output = try? model.prediction(image: resized,
                                                  iouThreshold: 0.45,
                                                  confidenceThreshold: 0.4) else { return }
        let ms = (CACurrentMediaTime() - t0) * 1000

        let dets = parseOutput(output)
        DispatchQueue.main.async {
            self.detections = dets
            self.inferenceMs = ms
        }
    }

    // MARK: - Output parsing
    // coordinates: (N, 4) normalized (cx, cy, w, h) in [0,1]
    // confidence:  (N, 80) class scores
    private func parseOutput(_ output: yolov8nOutput) -> [Detection] {
        let coords = output.coordinates
        let confs  = output.confidence
        let n = coords.shape[0].intValue

        var results: [Detection] = []
        results.reserveCapacity(n)

        for i in 0..<n {
            let cx = coords[[i, 0] as [NSNumber]].floatValue
            let cy = coords[[i, 1] as [NSNumber]].floatValue
            let w  = coords[[i, 2] as [NSNumber]].floatValue
            let h  = coords[[i, 3] as [NSNumber]].floatValue

            var maxConf: Float = 0
            var classIdx = 0
            for j in 0..<80 {
                let c = confs[[i, j] as [NSNumber]].floatValue
                if c > maxConf { maxConf = c; classIdx = j }
            }

            let rect = CGRect(
                x:      CGFloat(cx - w / 2),
                y:      CGFloat(cy - h / 2),
                width:  CGFloat(w),
                height: CGFloat(h)
            )
            results.append(Detection(classIndex: classIdx, confidence: maxConf, rect: rect))
        }
        return results.sorted { $0.confidence > $1.confidence }
    }

    // MARK: - Resize
    private func resize(_ buffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        var out: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(size.width), Int(size.height),
                            kCVPixelFormatType_32BGRA, nil, &out)
        guard let out else { return nil }
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let scaleX = size.width  / CGFloat(CVPixelBufferGetWidth(buffer))
        let scaleY = size.height / CGFloat(CVPixelBufferGetHeight(buffer))
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        CIContext().render(scaled, to: out)
        return out
    }
}
