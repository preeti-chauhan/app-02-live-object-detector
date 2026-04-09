# Live Object Detector

Object detection on iPhone using DETR and YOLOv8 — bounding box prediction with class labels and confidence scores, exported to CoreML for on-device inference.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_detr_architecture.ipynb` | DETR internals: encoder-decoder transformer, bipartite matching, Hungarian algorithm |
| `02_yolov8_inference.ipynb` | YOLOv8 on COCO — anchor-free detection, mAP evaluation, IoU |
| `03_detection_visualization.ipynb` | Bounding box visualization, attention maps, NMS explained |
| `04_coreml_export.ipynb` | Export YOLOv8 to CoreML, latency benchmark on Apple Neural Engine |

---

## Results

### DETR — Transformer-Based Detection

DETR encodes the image with a ResNet-50 backbone, then uses a transformer encoder-decoder with 100 learned object queries to predict bounding boxes directly — no anchors, no NMS.

<img src="assets/detr_detection.png"/>

**Cross-attention maps** show which image regions each object query attends to when predicting its box. Each query specializes to a different spatial area:

<img src="assets/detr_attention.png"/>

---

### YOLOv8 — Anchor-Free CNN Detection

YOLOv8 divides the image into a grid and predicts boxes at each cell using an anchor-free detection head. NMS filters overlapping predictions.

<img src="assets/yolo_detection.png"/>

**DETR vs YOLOv8 side-by-side** on the same image — same objects detected, different architectures:

<img src="assets/detr_vs_yolo.png"/>

---

### Detection Concepts

**Confidence thresholds** control how many boxes are shown. Lower threshold = more boxes but more noise:

<img src="assets/confidence_thresholds.png"/>

**Non-Maximum Suppression (NMS)** removes duplicate overlapping boxes, keeping only the highest-confidence detection per object:

<img src="assets/nms_comparison.png"/>

**COCO class distribution** across a sample of images — person is by far the most common class:

<img src="assets/class_distribution.png"/>

---

### CoreML Export — Latency Benchmark

YOLOv8n exported to CoreML (6.5 MB). Benchmarked across compute unit configurations on Apple Silicon:

<img src="assets/yolo_coreml_benchmark.png"/>

| Compute Unit | Mean Latency |
|---|---|
| ALL (Neural Engine) | **4.2 ± 0.2 ms** |
| CPU_AND_NE | 4.3 ± 0.4 ms |
| CPU_ONLY | 17.0 ± 0.2 ms |

`ALL` routes to the Neural Engine automatically — ~4× faster than CPU-only. This is the config used in the iPhone app.

---

## iPhone App

Object detection running on-device via CoreML. Select a photo — the model draws bounding boxes with class labels and confidence scores.

---

## Dataset

**COCO 2017 (Common Objects in Context)**

80 everyday object classes — people, animals, vehicles, food, furniture, electronics, and more.

- 118k training images, 5k validation images
- Each image contains multiple objects, each annotated with a bounding box and class label
- Annotation format: `[x, y, width, height]` (top-left corner + size)

Unlike scene classification (one label per image), COCO requires the model to find *where* each object is and *what* it is — simultaneously.

Both DETR and YOLOv8 are pretrained on COCO, so no training from scratch is needed.

---

## Models

**DETR (Detection Transformer)** — Facebook AI, 2020
- First end-to-end object detector using a pure transformer
- No anchors, no NMS (non-maximum suppression) — uses bipartite matching to assign predictions to objects
- Encodes the image with a CNN backbone, then uses a transformer encoder-decoder to output a fixed set of bounding boxes
- Slower but architecturally elegant — connects directly to what you learned in app-01

**YOLOv8** — Ultralytics, 2023
- State-of-the-art real-time detector — "You Only Look Once"
- CNN-based with anchor-free detection head
- Extremely fast — designed for on-device inference
- Industry standard for production CV systems

| | DETR | YOLOv8 |
|---|---|---|
| Architecture | Transformer | CNN |
| Speed | Slower | Fast (real-time) |
| CoreML export | Complex | Simple |
| Learning value | High (builds on app-01) | High (industry standard) |

**Approach:** study DETR in notebooks to understand transformer-based detection, use YOLOv8 for the iPhone app.

---

## Technologies

| Technology | Used For |
|---|---|
| PyTorch + transformers | DETR architecture and inference |
| Ultralytics YOLOv8 | Training and fine-tuning |
| coremltools | CoreML export |
| SwiftUI + PhotosUI | iOS app UI |
| CoreML + Vision | On-device inference |
