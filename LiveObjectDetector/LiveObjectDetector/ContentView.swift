import SwiftUI
import PhotosUI

struct ContentView: View {

    @State private var selectedPhoto: PhotosPickerItem?
    @State private var displayImage: Image?
    @State private var uiImage: UIImage?
    @State private var detections: [Detection] = []
    @State private var isDetecting = false

    private let detector = DetectorService()

    // Consistent box colors indexed by class — cycles through 8 hues
    private let boxColors: [Color] = [
        .blue, .green, .orange, .pink, .purple, .red, .teal, .yellow
    ]

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                let w = geo.size.width - 32
                ScrollView {
                    VStack(spacing: 20) {

                        // ── Photo picker ──────────────────────────────────
                        PhotosPicker(selection: $selectedPhoto, matching: .images) {
                            if let displayImage {
                                displayImage
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: w, height: w)
                                    .clipped()
                                    .cornerRadius(16)
                                    .overlay(alignment: .bottomTrailing) {
                                        Label("Change", systemImage: "photo.badge.plus")
                                            .font(.caption)
                                            .fontWeight(.semibold)
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 6)
                                            .background(.ultraThinMaterial)
                                            .cornerRadius(20)
                                            .padding(10)
                                    }
                            } else {
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color(.systemGray6))
                                    .frame(width: w, height: w)
                                    .overlay {
                                        VStack(spacing: 8) {
                                            Image(systemName: "photo.badge.plus")
                                                .font(.system(size: 44))
                                                .foregroundStyle(.secondary)
                                            Text("Tap to choose a photo")
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                            }
                        }
                        .onChange(of: selectedPhoto) { _, newItem in
                            Task {
                                if let data = try? await newItem?.loadTransferable(type: Data.self),
                                   let ui = UIImage(data: data) {
                                    uiImage = ui
                                    displayImage = Image(uiImage: ui)
                                    detections = []
                                    runDetection(ui)
                                }
                            }
                        }

                        // ── Detection overlay ─────────────────────────────
                        if let displayImage, !detections.isEmpty {
                            ZStack {
                                displayImage
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: w, height: w)
                                    .clipped()

                                Canvas { ctx, size in
                                    for det in detections.prefix(15) {
                                        let r = det.rect
                                        let box = CGRect(
                                            x: r.minX * size.width,
                                            y: r.minY * size.height,
                                            width: r.width * size.width,
                                            height: r.height * size.height
                                        )
                                        let color = boxColors[det.classIndex % boxColors.count]
                                        ctx.stroke(Path(box), with: .color(color), lineWidth: 2)

                                        // Label background + text
                                        let label = "\(det.label) \(Int(det.confidence * 100))%"
                                        var text = AttributedString(label)
                                        text.font = .systemFont(ofSize: 11, weight: .semibold)
                                        text.foregroundColor = .white
                                        let resolved = ctx.resolve(Text(text))
                                        let textSize = resolved.measure(in: CGSize(width: 200, height: 20))
                                        let labelRect = CGRect(
                                            x: box.minX,
                                            y: max(0, box.minY - 18),
                                            width: textSize.width + 6,
                                            height: 18
                                        )
                                        ctx.fill(Path(labelRect), with: .color(color))
                                        ctx.draw(resolved, at: CGPoint(x: labelRect.minX + 3,
                                                                        y: labelRect.minY + 2))
                                    }
                                }
                                .frame(width: w, height: w)
                            }
                            .cornerRadius(16)
                        }

                        // ── Detect button ─────────────────────────────────
                        if let image = uiImage {
                            Button {
                                runDetection(image)
                            } label: {
                                HStack {
                                    if isDetecting {
                                        ProgressView().tint(.white).padding(.trailing, 4)
                                    }
                                    Text(isDetecting ? "Detecting…" : "Detect Objects")
                                        .fontWeight(.semibold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundStyle(.white)
                                .cornerRadius(12)
                            }
                            .disabled(isDetecting)
                        }

                        // ── Results ───────────────────────────────────────
                        if !detections.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Detected Objects (\(detections.count))")
                                    .font(.headline)

                                ForEach(Array(detections.prefix(10).enumerated()),
                                        id: \.offset) { index, det in
                                    DetectionRow(detection: det,
                                                 rank: index + 1,
                                                 color: boxColors[det.classIndex % boxColors.count])
                                }

                                if detections.count > 10 {
                                    Text("+ \(detections.count - 10) more")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                Divider()

                                Label("Detects 80 COCO object classes.", systemImage: "info.circle")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(16)
                        }

                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                }
            }
            .navigationTitle("Live Object Detector")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func runDetection(_ image: UIImage) {
        isDetecting = true
        detector.detect(image: image) { results in
            self.detections = results
            self.isDetecting = false
        }
    }
}

// ── Detection row ─────────────────────────────────────────────────────────────
struct DetectionRow: View {
    let detection: Detection
    let rank: Int
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Circle()
                    .fill(color)
                    .frame(width: 10, height: 10)
                Text("\(rank). \(detection.label.capitalized)")
                    .fontWeight(rank == 1 ? .bold : .regular)
                Spacer()
                Text(String(format: "%.1f%%", detection.confidence * 100))
                    .foregroundStyle(rank == 1 ? color : .secondary)
                    .fontWeight(rank == 1 ? .bold : .regular)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 6)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(rank == 1 ? color : Color.secondary)
                        .frame(width: geo.size.width * CGFloat(detection.confidence), height: 6)
                }
            }
            .frame(height: 6)
        }
    }
}

#Preview {
    ContentView()
}
