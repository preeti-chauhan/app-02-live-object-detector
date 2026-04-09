# Live Object Detector

Object detection on iPhone using DETR and YOLOv8 — bounding box prediction with class labels and confidence scores, exported to CoreML for on-device inference.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_detr_architecture.ipynb` | DETR internals: encoder-decoder transformer, bipartite matching, Hungarian algorithm |
| `02_yolov8_finetune.ipynb` | YOLOv8 on COCO — fine-tuning, mAP evaluation, IoU |
| `03_detection_visualization.ipynb` | Bounding box visualization, attention maps, NMS explained |
| `04_coreml_export.ipynb` | Export YOLOv8 to CoreML, latency benchmark on Apple Neural Engine |

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

## Technologies

| Technology | Used For |
|---|---|
| PyTorch + transformers | DETR architecture and inference |
| Ultralytics YOLOv8 | Training and fine-tuning |
| coremltools | CoreML export |
| SwiftUI + PhotosUI | iOS app UI |
| CoreML + Vision | On-device inference |
