
# AI-Powered Breadboard Circuit Verification System

> **Current stage: Stage 1 — Component Detection only**

---

## Project Overview

This project builds a computer vision pipeline that will eventually verify whether a breadboard circuit is wired correctly by analysing a camera image.

The system is developed in stages. Only **Stage 1** is currently implemented.

---

## Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| **1** | **Component detection** | ✅ Active |
| 2 | Breadboard detection & perspective correction | 🔜 Future |
| 3 | Breadboard grid / coordinate mapping | 🔜 Future |
| 4 | Wire endpoint detection | 🔜 Future |
| 5 | Circuit graph reconstruction | 🔜 Future |
| 6 | Verification against predefined circuits | 🔜 Future |
| 7 | Frontend dashboard with visual overlays | 🔜 Future |

---

## Stage 1: Component Detection

Train and run a YOLO-based object detection model that identifies electronic components in breadboard images.

**Target classes (default):**
- `resistor`
- `LED`
- `jumper_wire`
- `pushbutton`
- `potentiometer`
- `capacitor`
- `breadboard`

> The actual classes are defined by your dataset's `data.yaml`. Edit that file to match your labels.

---

## Project Structure

```
project/
├── datasets/
│   ├── data.yaml          ← dataset config (edit this)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/              (optional)
│       ├── images/
│       └── labels/
├── models/
│   └── best.pt            ← best trained weights (auto-copied here after training)
├── runs/                  ← YOLO training run outputs
├── src/
│   ├── train.py           ← training script
│   ├── detect.py          ← inference script
│   └── utils.py           ← shared helper functions
├── outputs/               ← annotated detection images
├── test_images/           ← put your test images here
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> GPU support: if you have an NVIDIA GPU, install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Dataset Format

The training script expects a YOLO-format dataset:

```
datasets/
  data.yaml          ← class names + split paths
  train/
    images/          ← .jpg / .png images
    labels/          ← .txt files (one per image, YOLO format)
  valid/
    images/
    labels/
  test/              ← optional
    images/
    labels/
```

Each `.txt` label file contains one row per object:

```
<class_id> <x_centre> <y_centre> <width> <height>
```

All values normalised to `[0, 1]` relative to image dimensions.

Edit `datasets/data.yaml` to match your dataset's class names and split paths.

---

## Training

```bash
python src/train.py \
    --data   datasets/data.yaml \
    --model  yolov8n.pt \
    --epochs 50 \
    --imgsz  640 \
    --output runs/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `datasets/data.yaml` | Dataset YAML path |
| `--model` | `yolov8n.pt` | Pretrained base weights |
| `--epochs` | `50` | Training epochs |
| `--imgsz` | `640` | Input image size (px, square) |
| `--batch` | `16` | Batch size |
| `--output` | `runs/` | Training output directory |
| `--name` | `component_detection` | Run name |
| `--device` | *(auto)* | `cpu`, `0`, `0,1`, etc. |

After training, the best weights are automatically copied to `models/best.pt`.

---

## Detection

```bash
python src/detect.py \
    --weights models/best.pt \
    --source  test_images/ \
    --output  outputs/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `models/best.pt` | Trained weights path |
| `--source` | *(required)* | Image file or directory |
| `--output` | `outputs/` | Save annotated images here |
| `--conf` | `0.25` | Confidence threshold |
| `--imgsz` | `640` | Inference image size |
| `--device` | *(auto)* | Device override |
| `--save-txt` | off | Also save `.txt` result summaries |

---

## Output

For each input image the detection script:

1. Saves an **annotated copy** (with coloured bounding boxes and labels) to `outputs/`.
2. Prints a **detection table** to stdout:

```
  Detections for: circuit_01.jpg
    [01] resistor             conf=91.23%  box=(142,88) → (198,134)
    [02] LED                  conf=87.65%  box=(310,201) → (342,255)
    [03] jumper_wire          conf=73.40%  box=(55,300) → (420,318)
```

If `--save-txt` is passed, a matching `.txt` file is saved next to each output image.

---

## Notes

- **Model size:** `yolov8n` (nano) is used by default — fast to train and run, good for prototyping. Upgrade to `yolov8s` / `yolov8m` for better accuracy once the pipeline is validated.
- **Epochs:** 50 epochs is a reasonable starting point. Increase to 100–200 for a production-quality model.
- **Dataset size:** YOLO can start learning from ~200–500 images per class; more is always better.


