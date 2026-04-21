import CoreML
import UIKit

struct Detection {
    let classIndex: Int
    let confidence: Float
    let rect: CGRect  // normalized [0,1] x, y, width, height (origin at top-left)

    var label: String { DetectorService.classes[classIndex] }
}

class DetectorService {

    private let model: yolov8n

    // COCO 80 classes — order matches YOLOv8 training labels
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

    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let m = try? yolov8n(configuration: config) else {
            fatalError("Failed to load yolov8n.mlpackage")
        }
        self.model = m
    }

    /// Detect objects in a UIImage and return results sorted by confidence.
    func detect(image: UIImage, completion: @escaping ([Detection]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let buffer = self.pixelBuffer(from: image),
                  let output = try? self.model.prediction(
                      image: buffer,
                      iouThreshold: 0.45,
                      confidenceThreshold: 0.4) else {
                DispatchQueue.main.async { completion([]) }
                return
            }
            let detections = self.parseOutput(output)
            DispatchQueue.main.async { completion(detections) }
        }
    }

    // MARK: - Private helpers

    /// Resize UIImage to 640×640 and convert to CVPixelBuffer (BGRA).
    /// Equivalent to YOLOv8's default letterbox resize used during export.
    private func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        // Draw UIImage (handles any orientation) into a 640×640 context
        let side = CGSize(width: 640, height: 640)
        UIGraphicsBeginImageContextWithOptions(side, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: side))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let cgImage = resized?.cgImage else { return nil }

        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, 640, 640,
                                  kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess,
              let pb = buffer else { return nil }

        CVPixelBufferLockBaseAddress(pb, [])
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: 640, height: 640,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(pb, [])
            return nil
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: 640, height: 640))
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }

    /// Parse CoreML output arrays into Detection structs.
    /// coordinates: (N, 4) normalized (cx, cy, w, h) in [0, 1]
    /// confidence:  (N, 80) class probabilities
    private func parseOutput(_ output: yolov8nOutput) -> [Detection] {
        let coords = output.coordinates
        let confs  = output.confidence
        let n = coords.shape[0].intValue

        var detections: [Detection] = []
        detections.reserveCapacity(n)

        for i in 0..<n {
            let cx = coords[[i, 0] as [NSNumber]].floatValue
            let cy = coords[[i, 1] as [NSNumber]].floatValue
            let w  = coords[[i, 2] as [NSNumber]].floatValue
            let h  = coords[[i, 3] as [NSNumber]].floatValue

            // argmax over 80 class scores
            var maxConf: Float = 0
            var classIdx = 0
            for j in 0..<80 {
                let c = confs[[i, j] as [NSNumber]].floatValue
                if c > maxConf { maxConf = c; classIdx = j }
            }

            // Convert (cx, cy, w, h) → (x, y, width, height) with top-left origin
            let rect = CGRect(
                x:      CGFloat(cx - w / 2),
                y:      CGFloat(cy - h / 2),
                width:  CGFloat(w),
                height: CGFloat(h)
            )
            detections.append(Detection(classIndex: classIdx, confidence: maxConf, rect: rect))
        }

        return detections.sorted { $0.confidence > $1.confidence }
    }
}
