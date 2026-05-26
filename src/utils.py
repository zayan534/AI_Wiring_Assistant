"""
Utility functions for the AI-Powered Breadboard Circuit Verification System
Stage 1: Component Detection
"""

import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union


# ──────────────────────────────────────────────
# Path Helpers
# ──────────────────────────────────────────────

def check_path_exists(path: Union[str, Path], label: str = "Path") -> Path:
    """Verify a path exists; raise FileNotFoundError with a clear message if not."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p.resolve()}")
    return p


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────
# Dataset Helpers
# ──────────────────────────────────────────────

def load_class_names(data_yaml_path: Union[str, Path]) -> list[str]:
    """
    Parse class names from a YOLO data.yaml file.

    Expects a YAML file with a 'names' key that maps to either:
      - A list:  names: [resistor, LED, ...]
      - A dict:  names: {0: resistor, 1: LED, ...}
    """
    yaml_path = check_path_exists(data_yaml_path, "data.yaml")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names")
    if names is None:
        raise KeyError(f"'names' key not found in {yaml_path}")

    if isinstance(names, dict):
        # Sort by integer key to preserve class-index order
        names = [names[k] for k in sorted(names.keys())]

    return list(names)


def validate_dataset_structure(dataset_root: Union[str, Path]) -> bool:
    """
    Check that the dataset directory has the expected YOLO layout.
    Returns True if structure is valid, prints warnings otherwise.
    """
    root = Path(dataset_root)
    ok = True

    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            p = root / split / sub
            if not p.exists():
                print(f"  [WARNING] Missing expected folder: {p}")
                ok = False

    yaml_file = root / "data.yaml"
    if not yaml_file.exists():
        print(f"  [WARNING] data.yaml not found at: {yaml_file}")
        ok = False

    return ok


# ──────────────────────────────────────────────
# Detection Formatting
# ──────────────────────────────────────────────

def format_detections(results, class_names: Optional[list[str]] = None) -> list[dict]:
    """
    Convert Ultralytics Results object into a list of plain dicts.

    Each dict contains:
      - class_id   (int)
      - class_name (str)
      - confidence (float, rounded to 4 dp)
      - bbox       (dict with x1, y1, x2, y2 in pixel coords)
    """
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        # Resolve class names from result if not supplied
        names = class_names or result.names

        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id] if isinstance(names, (list, dict)) else str(cls_id)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                }
            )

    return detections


def print_detections(detections: list[dict], image_name: str = "") -> None:
    """Pretty-print a list of detection dicts to stdout."""
    header = f"  Detections for: {image_name}" if image_name else "  Detections:"
    print(header)

    if not detections:
        print("    (none)")
        return

    for i, det in enumerate(detections, 1):
        bb = det["bbox"]
        print(
            f"    [{i:02d}] {det['class_name']:<20s} "
            f"conf={det['confidence']:.2%}  "
            f"box=({bb['x1']:.0f},{bb['y1']:.0f}) → ({bb['x2']:.0f},{bb['y2']:.0f})"
        )


# ──────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────

def save_annotated_image(
    results,
    output_path: Union[str, Path],
    image_name: Optional[str] = None,
) -> Path:
    """
    Save a YOLO annotated result image (with bounding boxes drawn) to disk.

    Ultralytics Results.plot() returns a BGR numpy array; we write it with cv2.
    """
    out = Path(output_path)
    ensure_dir(out.parent)

    if image_name:
        out = out.parent / image_name if out.suffix == "" else out

    # results is expected to be a single Result object here
    annotated = results.plot()  # BGR ndarray
    cv2.imwrite(str(out), annotated)
    return out


def collect_image_paths(source: Union[str, Path]) -> list[Path]:
    """
    Return a list of image file paths from:
      - A single image file
      - A directory (recursively finds common image extensions)
    """
    source = Path(source)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    if source.is_file():
        if source.suffix.lower() in extensions:
            return [source]
        raise ValueError(f"File is not a recognised image format: {source}")

    if source.is_dir():
        found = sorted(
            p for p in source.rglob("*") if p.suffix.lower() in extensions
        )
        if not found:
            raise FileNotFoundError(f"No images found in directory: {source}")
        return found

    raise FileNotFoundError(f"Source not found: {source}")
