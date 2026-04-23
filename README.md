# 🛸 Vision-Based Anti-Drone Detection using YOLOv8

> A robust multi-class aerial object detection framework based on **YOLOv8s** with **hard negative learning** and **dataset-level optimization** — built as a major project at NIT Calicut.

---

## 📋 Overview

Detecting drones in real-world surveillance environments is challenging due to small object size, motion blur, occlusion, and visual similarity with birds and airplanes. Most existing approaches fail to generalize because they train on narrow, single-class datasets and never explicitly teach the model what **not** to flag.

This project takes a data-centric angle:

- **Fused multi-source dataset** — Kaggle UAV drone images + COCO 2017 bird & airplane samples
- **Hard negative learning** — birds and airplanes are explicitly labeled classes, not background noise
- **Domain-targeted augmentation** — low-light, motion blur, resolution degradation, and sensor noise
- **YOLOv8s backbone** — anchor-free, real-time, tuned for small object detection

**Result: Drone AP@50 = 0.923 · mAP@50 = 0.775 · Bird→Drone confusion = 1% · ~84 FPS on Tesla T4**

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| Drone AP@50 | **0.923** |
| mAP@50 (all classes) | **0.775** |
| mAP@50-95 | **0.537** |
| Drone Precision | **0.970** |
| Drone Recall | **0.875** |
| Bird→Drone Confusion | **1%** |
| Inference Latency | **11.8 ms/image (~84 FPS)** |

### Per-Class Breakdown

| Class | Precision | Recall | AP@50 | mAP@50-95 |
|---|---|---|---|---|
| **Drone** | 0.970 | 0.875 | 0.923 | 0.617 |
| Bird | 0.651 | 0.486 | 0.613 | 0.409 |
| Airplane | 0.781 | 0.711 | 0.789 | 0.586 |
| **All (mean)** | **0.801** | **0.691** | **0.775** | **0.537** |

---

## 📁 Repository Structure

```
Drone_Detection_YOLO/
│
└── End_Sem.ipynb          # Full pipeline: data prep → training → evaluation → inference
```

The notebook is self-contained and designed to run on **Google Colab** with a GPU runtime.

---

## 🔧 Pipeline

```
Data Collection  →  Data Fusion  →  Augmentation  →  Model Training  →  Inference
   (Phase 1)          (Phase 2)        (Phase 3)         (Phase 4)        (Phase 5)
```

### Phase 1 — Data Collection
- **Drone:** [Kaggle UAV Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) — 1,359 annotated images (DJI Phantom, Mavic, Spark, hexarotors, fixed-wing UAVs)
- **Bird & Airplane:** COCO 2017 via FiftyOne — up to 1,500 combined images (category IDs 16 and 5)

### Phase 2 — Data Fusion & Class Mapping
All sources are merged into a unified YOLO-format dataset with the following class indices:

| Class | Index |
|---|---|
| drone | 0 |
| bird | 1 |
| airplane | 2 |

Dataset split: **80% train / 20% validation**. A **3× offline augmentation** expansion yields a final training set of ~6,630 images. The validation set (647 images, 1,047 instances) is kept augmentation-free.

### Phase 3 — Augmentation Pipeline
Beyond Ultralytics' built-in mosaic and mixup, four domain-targeted transforms are applied using [Albumentations](https://albumentations.ai/):

| Transform | Parameters | Probability |
|---|---|---|
| Low-light simulation | Brightness ∈ [0.3, 0.7] | p = 0.40 |
| Motion blur | Kernel ∈ {3, 5, 7} | p = 0.30 |
| Resolution degradation | Bicubic downsample 0.4–0.7× | p = 0.35 |
| Gaussian noise | σ ∈ [5, 25] | p = 0.30 |

Plus Ultralytics built-ins: mosaic (1.0), mixup (0.15), copy-paste (0.1), HSV shifts, rotation ±10°.

### Phase 4 — Model Training
- **Architecture:** YOLOv8s (Ultralytics 8.4.35)
- **Framework:** PyTorch 2.10.0+cu128
- **Epochs:** 50 · **Batch:** 16 · **Input:** 640×640
- **Optimizer:** SGD, momentum 0.937, cosine annealing LR
- **GPU:** Tesla T4 (14.9 GiB)

### Phase 5 — Inference with Edge-Case Preprocessing

Two optional preprocessing helpers handle real-world edge cases before inference:

**Low-light (CLAHE):** Converts to LAB color space → applies CLAHE on the L channel (clipLimit=3.0, tileGrid=8×8) → reconstructs BGR.

**Low-resolution (upscaling):** If `max(h, w) < 640px`, upscales using `INTER_CUBIC` to avoid aliasing artifacts from long-range drone images.

```python
save_dir = run_inference(
    "path/to/image.jpg",
    conf=0.25,
    enhance_low_light=True,   # set False for well-lit images
    enhance_low_res=False,
)
```

---

## 💡 Hard Negative Learning

The core contribution of this project is treating **birds and airplanes as explicitly labeled classes** rather than background noise during training.

**Without hard negatives:** the model only learns "drone vs. nothing." Birds share similar size, silhouette, and sky background with drones — causing frequent false alarms in single-class systems.

**With hard negatives:** the C2f feature extractor is forced to learn discriminative representations that separate all three classes in feature space. At inference time, a region confidently labeled `bird` cannot simultaneously activate the `drone` class — directly suppressing false positives.

The result: **bird-to-drone confusion of only 1%**, confirming the strategy works.

---

## 🚀 Quick Start

### 1. Open in Colab

Click the badge or open `End_Sem.ipynb` directly in Google Colab. Make sure to select a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

### 2. Install Dependencies

The notebook installs all required packages automatically:

```bash
pip install ultralytics albumentations kaggle Pillow matplotlib PyYAML fiftyone
```

### 3. Set up Kaggle API

Upload your `kaggle.json` credentials file when prompted, or mount your Google Drive and copy it to `~/.kaggle/kaggle.json`.

### 4. Run All Cells

The notebook runs end-to-end:
1. Downloads datasets
2. Builds the unified YOLO dataset
3. Applies augmentation
4. Trains YOLOv8s for 50 epochs
5. Evaluates and plots metrics
6. Runs inference with optional preprocessing

### 5. Download Outputs

The notebook downloads `best.pt` (trained weights) and the unified dataset zip automatically via `files.download()`.

---

## 📊 Ablation Study

| Configuration | Drone AP@50 | mAP@50 | mAP@50-95 |
|---|---|---|---|
| (A) Drone-only, no hard negatives | 0.932 | 0.694 | 0.458 |
| (B) Fused dataset, no augmentation | 0.918 | 0.751 | 0.512 |
| **(C) Full framework (proposed)** | **0.923** | **0.775** | **0.537** |

Dataset fusion alone contributes **+5.7% mAP@50** — the single largest gain, confirming that class diversity is the dominant generalization factor. The augmentation pipeline adds a further **+2.4% mAP@50**.

Note: the marginal drone AP@50 decrease from (A) to (C) is expected — representational capacity is now shared across three classes. Critically, configuration (A) produces significant bird-as-drone false positives that are invisible in single-class AP metrics but are critical deployment failures.

---

## 🎯 Deployment Scenarios & Confidence Thresholds

| Scenario | Primary Need | Recommended Confidence |
|---|---|---|
| Airport perimeter | Low false positive rate | 0.60 – 0.75 |
| Critical infrastructure | Multi-class discrimination | 0.40 – 0.60 |
| Prison environments | Low-light robustness | 0.25 – 0.40 |
| Military forward base | Real-time edge inference | 0.10 – 0.20 |

---

## ⚙️ Model Statistics

| Parameter | Value |
|---|---|
| Architecture | YOLOv8s |
| Parameters | 11,126,745 |
| GFLOPs | 28.4 |
| Preprocessing | 1.8 ms/image |
| Inference | 8.0 ms/image |
| Post-processing | 2.0 ms/image |
| **Total latency** | **11.8 ms/image** |
| Frame rate | **~84 FPS (Tesla T4)** |

---

## 📚 Dataset Sources

| Dataset | Source | Link |
|---|---|---|
| Kaggle UAV Detection | Muki2003 | [kaggle.com/datasets/muki2003/yolo-drone-detection-dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) |
| COCO 2017 | Microsoft | [cocodataset.org](https://cocodataset.org/) |

---

## 👥 Authors

**Department of Electronics and Communication Engineering, NIT Calicut**

| Name | Roll No. |
|---|---|
| Yasaswi Akkineni | B231347EC |
| B. S. V. Hyndavi | B230537EC |
| U. Jaya Chandra | B230712EC |
| Harith Manoj | B230335EC |

**Project Guide:** Dr. Karthik Rudramuni


## 📝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Albumentations](https://albumentations.ai/) for the augmentation pipeline
- [FiftyOne](https://voxel51.com/fiftyone/) for COCO dataset loading
- AI-assisted tools (ChatGPT by OpenAI,Claude by Anthropic) were used for language refinement during manuscript preparation. All technical content, experiments, results, and conclusions were independently developed and validated by the authors.
