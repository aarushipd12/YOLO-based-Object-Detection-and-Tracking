# Drone Footage Object Detection & Tracking

A deep learning pipeline for detecting, tracking, and alerting on objects in aerial drone footage — built on YOLOv8 / YOLOv8-World and trained on the VisDrone dataset. Supports fixed-class detection, open-vocabulary text-prompt detection, low-light inference, and a restricted-zone alert system.

**🔗 Live demo:** [dashboarddronefootageobjectdetection.streamlit.app](https://dashboarddronefootageobjectdetection.streamlit.app/)
**🔗 Dashboard/deployment repo:** [drone-detection-dashboard](https://github.com/aarushipd12/DASHBOARD---Drone-Footage-Object-Detection-and-Tracking) 

---

## Overview

Aerial drone footage presents a harder detection problem than ground-level imagery: objects are small relative to the frame, viewing angles are steep and inconsistent, and footage often spans a wide range of lighting conditions in a single flight. This project addresses that with a YOLOv8-based pipeline specifically tuned for aerial perspectives, using [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference) to recover small-object recall that a single full-frame forward pass typically loses, plus a YOLO-World variant for open-vocabulary detection when the target classes aren't known in advance.

## Features

- **General object detection** — cars, pedestrians, trucks, and other common classes from aerial viewpoints, trained on VisDrone.
- **Night-time detection** — evaluated specifically on low-light drone footage rather than assuming daytime-only performance.
- **Prompt-based detection** — open-vocabulary detection via YOLO-World, where a user supplies free-text class names (e.g. `"yellow car, black car"`) at inference time instead of being limited to the fixed training label set.
- **Smart restricted-zone alert system** — flags any person detected within a user-defined bounding region and triggers an alert, aimed at a real surveillance/security use case rather than detection in isolation.
- **Image and video support** across all of the above, including SAHI-sliced inference and prompt-based tracking on video.

## Dataset

Trained on the [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset), a benchmark built specifically for aerial computer vision tasks (drone-captured imagery across varied altitudes, angles, and densities).

- **Images** — organized into `train/`, `val/`, `test/` splits, each with raw images and matching annotation files.
- **Video** — organized into sequence folders, each containing extracted frame subfolders plus per-frame bounding box/class annotations.

VisDrone annotations were parsed and converted into YOLO format as part of the training pipeline (see `training/` notebook).

## Model & Pipeline

| Stage | Approach |
|---|---|
| Detection backbone | YOLOv8 / YOLOv8-World |
| Small-object recall | SAHI sliced inference (tiled forward passes over high-resolution frames) |
| Open-vocabulary detection | YOLO-World text-prompt conditioning |
| Alert logic | Custom zone-intersection check on detected person bounding boxes |

**Pipeline stages:**
1. Parse and convert VisDrone annotations to YOLO format
2. Train on image and video data (`Training/`)
3. Evaluate on a held-out VisDrone test split (`Test/`)
4. Run inference — standard, SAHI-sliced, and text-prompt-based (`Inference/`)
5. Apply the restricted-zone alert module on top of detections (`Bonus Features/`)

## Project Structure

```
YOLO-based-Object-Detection-and-Tracking/
│
├── Bonus Features/
│   └── alert_system.ipynb                    # Restricted-zone alert system
│
├── Inference/
│   ├── SAHI_inference.ipynb                  # YOLO + SAHI sliced inference pipeline
│   ├── Text_prompt_inference_on_video.ipynb   # Prompt-based detection on video
│   └── text_prompt_inference_on_images.ipynb  # Prompt-based detection on images
│
├── Results/
│   ├── BoxF1_curve.png                 # F1 score vs. confidence
│   ├── BoxPR_curve.png                 # Precision-Recall curve
│   ├── BoxP_curve.png                  # Precision vs. confidence
│   ├── BoxR_curve.png                  # Recall vs. confidence
│   ├── labels.jpg                      # Dataset label distribution
│   └── training_results_summary.png    # Training loss/metric curves
│
├── Test/
│   └── test_evaluation.ipynb           # Evaluation on held-out test set
│
├── Training/
│   └── yolov8s_world_training.ipynb    # YOLOv8/YOLOv8-World training notebook
│
├── Project report/
│   ├── AIML-012_EndEval_FINAL_REPORT.pdf
│   └── MidEval.pptx
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/aarushipd12/YOLO-based-Object-Detection-and-Tracking.git
cd YOLO-based-Object-Detection-and-Tracking
pip install -r requirements.txt
```
Python 3.8+ recommended. A CUDA-capable GPU is strongly recommended for training and video inference — the notebooks will run on CPU but significantly slower.

## Usage

**1. Train the model** — open `Training/yolov8s_world_training.ipynb`, set the VisDrone dataset path, adjust epochs/batch size/image size, and run. Trained weights (`.pt`) save to your run directory.

**2. Evaluate on the test set** — open `Test/test_evaluation.ipynb`, point `model_path` and `data_path` at your trained weights and VisDrone test split, and run all cells to reproduce the metrics in `Results/`.

**3. Run inference:**
- *Standard / SAHI* — `Inference/SAHI_inference.ipynb`, set model + input paths.
- *Text-prompt (images)* — `Inference/text_prompt_inference_on_images.ipynb`, enter a comma-separated prompt (e.g. `"pedestrian, bicycle"`).
- *Text-prompt (video)* — `Inference/Text_prompt_inference_on_video.ipynb`, set video path + prompt; outputs an annotated, tracked video of only the prompted classes.

**4. Run the alert system** — `Bonus Features/alert_system.ipynb`, load an image/video, define a restricted zone by its top-left/bottom-right coordinates, and the system flags any person detected inside it.

**5. Or skip the notebooks entirely** — the [Streamlit dashboard](https://dashboarddronefootageobjectdetection.streamlit.app/) wraps all of the above into an interactive UI. See the [dashboard repo](https://github.com/aarushipd12) for local setup.

## Results

Evaluation curves from the held-out VisDrone test split are in `results/`: F1-vs-confidence, precision-recall, precision-vs-confidence, and recall-vs-confidence, alongside the training loss/metric curves and the dataset's label distribution.

> Add your actual headline numbers here once pulled from `test_evaluation.ipynb` / the PR curve — e.g. **mAP@0.5: __%, Precision: __%, Recall: __%** — a reviewer will look for these first, so don't leave this section as just filenames.

**Qualitative results** (see the live demo for interactive versions):
- General object detection across dense traffic/pedestrian scenes
- Pedestrian-specific detection in cluttered aerial views
- Night-time / low-light detection
- Prompt-based detection (e.g. `"yellow car, black car, white car"`, `"red car"` on video)
- Restricted-zone alert triggering on person detection

## Reports

Full project documentation is in `Project report/`:
- `AIML-012_EndEval_FINAL_REPORT.pdf` — final evaluation report
- `MidEval.pptx` — mid-evaluation slides

## Acknowledgements

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), and [SAHI](https://github.com/obss/sahi), trained on the [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset).