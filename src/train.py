"""
Stage 1 – Component Detection Training Script
AI-Powered Breadboard Circuit Verification System

Usage:
    python src/train.py \
        --data   datasets/data.yaml \
        --model  yolov8n.pt \
        --epochs 50 \
        --imgsz  640 \
        --output runs/
"""

import argparse
import sys
from pathlib import Path

# ── Make sure the project src/ folder is importable ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import check_path_exists, ensure_dir, load_class_names, validate_dataset_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for electronic component detection (Stage 1)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/data.yaml",
        help="Path to the YOLO dataset YAML file  (default: datasets/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO weights to fine-tune  (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs  (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square, pixels)  (default: 640)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/",
        help="Directory where training runs are saved  (default: runs/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="component_detection",
        help="Name for this training run  (default: component_detection)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on: '' = auto, 'cpu', '0', '0,1', etc.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size  (default: 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Stage 1 – Component Detection Training")
    print("  AI Breadboard Circuit Verification System")
    print("═" * 60)

    # ── Validate inputs ───────────────────────────────────────────────────────
    print("\n[1/4] Validating inputs …")
    data_yaml = check_path_exists(args.data, "Dataset YAML")

    dataset_root = data_yaml.parent
    print(f"  Dataset root : {dataset_root.resolve()}")
    print(f"  data.yaml    : {data_yaml.resolve()}")

    dataset_ok = validate_dataset_structure(dataset_root)
    if not dataset_ok:
        print("  [WARNING] Dataset structure has issues — training may still proceed.")

    class_names = load_class_names(data_yaml)
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")

    output_dir = ensure_dir(args.output)
    print(f"  Output dir   : {output_dir.resolve()}")

    # ── Import YOLO (deferred so errors surface cleanly) ──────────────────────
    print("\n[2/4] Loading YOLO model …")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics is not installed.")
        print("  Run:  pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    print(f"  Base weights : {args.model}")
    print(f"  Architecture : {model.info(verbose=False)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Starting training …")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Image   : {args.imgsz}px")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {'auto' if args.device == '' else args.device}")
    print()

    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(output_dir),
        device=args.device if args.device else None,
        exist_ok=False,
    )

    # ── Save best weights to models/ ──────────────────────────────────────────
    print("\n[4/4] Saving best weights …")
    models_dir = ensure_dir(Path("models"))

    best_src = Path(output_dir) / args.name / "weights" / "best.pt"
    best_dst = models_dir / "best.pt"

    if best_src.exists():
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"  Best weights copied → {best_dst.resolve()}")
    else:
        print(f"  [WARNING] best.pt not found at {best_src} — check run output.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training complete!")
    print(f"  Run folder : {(Path(output_dir) / args.name).resolve()}")
    print(f"  Best model : {best_dst.resolve()}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
