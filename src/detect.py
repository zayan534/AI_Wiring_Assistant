"""
Stage 1 – Component Detection Inference Script
AI-Powered Breadboard Circuit Verification System

Usage:
    python src/detect.py \
        --weights models/best.pt \
        --source  test_images/ \
        --output  outputs/ \
        --conf    0.25
"""

import argparse
import sys
from pathlib import Path

# ── Make sure the project src/ folder is importable ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    check_path_exists,
    collect_image_paths,
    ensure_dir,
    format_detections,
    print_detections,
    save_annotated_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run component detection on images (Stage 1)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained YOLO weights  (default: models/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file or directory of images to run inference on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Directory to save annotated images  (default: outputs/)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (0–1)  (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square, pixels)  (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' = auto, 'cpu', '0', etc.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save detections as YOLO-format .txt label files",
    )
    return parser.parse_args()


def save_detections_txt(detections: list[dict], output_path: Path) -> None:
    """Save detection results as a plain text summary alongside the image."""
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"Detections for: {output_path.name}\n")
        f.write(f"Total: {len(detections)}\n\n")
        for i, det in enumerate(detections, 1):
            bb = det["bbox"]
            f.write(
                f"[{i:02d}] class={det['class_name']}  "
                f"conf={det['confidence']:.4f}  "
                f"bbox=({bb['x1']:.1f},{bb['y1']:.1f},{bb['x2']:.1f},{bb['y2']:.1f})\n"
            )


def main() -> None:
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Stage 1 – Component Detection Inference")
    print("  AI Breadboard Circuit Verification System")
    print("═" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading model …")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics is not installed.")
        print("  Run:  pip install ultralytics")
        sys.exit(1)

    weights_path = check_path_exists(args.weights, "Weights file")
    model = YOLO(str(weights_path))
    print(f"  Weights  : {weights_path.resolve()}")
    print(f"  Classes  : {list(model.names.values())}")
    print(f"  Conf thr : {args.conf}")
    print(f"  Img size : {args.imgsz}px")

    # ── Collect images ────────────────────────────────────────────────────────
    print("\n[2/3] Collecting images …")
    image_paths = collect_image_paths(args.source)
    print(f"  Found {len(image_paths)} image(s) in: {Path(args.source).resolve()}")

    output_dir = ensure_dir(args.output)
    print(f"  Output  : {output_dir.resolve()}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n[3/3] Running inference …\n")

    total_detections = 0

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False,
        )

        # Save annotated image
        out_img_path = output_dir / img_path.name
        save_annotated_image(results[0], out_img_path)

        # Parse and display detections
        detections = format_detections(results)
        print_detections(detections, image_name=img_path.name)

        if args.save_txt:
            save_detections_txt(detections, out_img_path)

        total_detections += len(detections)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * 60)
    print(f"  Processed : {len(image_paths)} image(s)")
    print(f"  Total detections : {total_detections}")
    print(f"  Annotated images saved to : {output_dir.resolve()}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
