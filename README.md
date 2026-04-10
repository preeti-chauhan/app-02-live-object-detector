# Live Object Detector

Object detection on iPhone using DETR and YOLOv8 — bounding box prediction with class labels and confidence scores, exported to CoreML for on-device inference.

---

## Dataset

**COCO 2017 (Common Objects in Context)**

80 everyday object classes — people, animals, vehicles, food, furniture, electronics, and more.

- 118k training images, 5k validation images
- Each image contains multiple objects, each annotated with a bounding box and class label
- Annotation format: `[x, y, width, height]` (top-left corner + size)

Unlike scene classification (one label per image), COCO requires the model to find *where* each object is and *what* it is — simultaneously.

Both DETR and YOLOv8 are pretrained on COCO — no training from scratch needed.

---

## Models

Two models are studied: DETR to understand transformer-based detection, YOLOv8 for deployment.

**DETR (Detection Transformer)** — Facebook AI, 2020
- First end-to-end object detector using a pure transformer
- No anchors, no NMS — uses bipartite matching to assign predictions to ground truth
- Encodes the image with a CNN backbone, then uses a transformer encoder-decoder with 100 learned object queries to predict boxes directly
- 41M parameters — too large for real-time on-device inference

**YOLOv8** — Ultralytics, 2023
- State-of-the-art real-time detector — "You Only Look Once"
- CNN-based with anchor-free detection head and NMS post-processing
- 3.2M parameters — designed for on-device inference
- Industry standard for production CV systems

| | DETR | YOLOv8n |
|---|---|---|
| Architecture | Transformer | CNN |
| Parameters | 41M | 3.2M |
| Speed | Slower | Fast (real-time) |
| CoreML export | Fails (dynamic control flow) | One line |
| Use in this project | Notebooks — architecture study | iPhone app — deployment |

**Why DETR can't export to CoreML:** At inference time, DETR's post-processing involves dynamic control flow — variable-length outputs and conditional masking based on confidence threshold cause `torch.jit.trace` to break, since trace records one fixed execution path and fails when output shapes change across inputs. YOLOv8 bakes NMS directly into the CoreML model as a static `NMSLayer`, giving it a fixed computation graph regardless of input.

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

**Sample image used for detection:**

<img src="assets/sample_image.jpeg" width="600"/>

---

### DETR — Transformer-Based Detection

<img src="assets/detr_detection.png"/>

**Cross-attention maps** show which image regions each object query attends to when predicting its box. Each query specializes to a different spatial area:

<img src="assets/detr_attention.png"/>

---

### YOLOv8 — Anchor-Free CNN Detection

<img src="assets/yolo_detection.png"/>

---

### DETR vs YOLOv8

Same image, both models. The panel titles show the key difference:

<img src="assets/detr_vs_yolo.png"/>

The chart below makes the difference clear:

<img src="assets/detr_vs_yolo_chart.png"/>

Red bars are classes DETR detects but YOLOv8 misses — handbag (pedestrian carrying a bag, low confidence), truck (the van), stop sign. DETR also fires 8 traffic light boxes vs YOLOv8's 3, over-detecting on lamp posts. YOLOv8 produces fewer, more precise detections with 13× fewer parameters.

Combined with a clean CoreML export, YOLOv8 is used for all practical inference. DETR is studied for its architecture — the first end-to-end detector with no anchors and no NMS.

---

### Detection Concepts

**Confidence thresholds** control how many boxes are shown. Lower threshold = more boxes but more noise:

<img src="assets/confidence_thresholds.png"/>

**Non-Maximum Suppression (NMS)** removes duplicate overlapping boxes, keeping only the highest-confidence detection per object:

<img src="assets/nms_comparison.png"/>

**Detection across diverse scenes** — YOLOv8n applied to varied real-world photos, showing the model generalizes across object types and contexts:

<img src="assets/multi_scene_detection.png"/>

**COCO class distribution** across the sample images — person and car dominate outdoor scenes, while indoor scenes surface food and tableware classes:

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

End-to-end inference confirmed: CoreML loads the `.mlpackage`, runs inference, and returns `coordinates` (N×4 normalized boxes) and `confidence` (N×80 class scores). Output format verified before building the iPhone app.

---

## iPhone App

Object detection running on-device via CoreML. Select a photo — the model draws bounding boxes with class labels and confidence scores.

---

## Technologies

| Technology | Used For |
|---|---|
| PyTorch + transformers | DETR architecture and inference |
| Ultralytics YOLOv8 | Detection, inference, CoreML export |
| coremltools | CoreML export and benchmarking |
| SwiftUI + PhotosUI | iOS app UI |
| CoreML | On-device inference |
