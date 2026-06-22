"""
src/train_animal_detector.py
STAGE 1: Train a YOLOv8 object-detection model that finds animals on the
road and classifies their species (Animal type categorization).

Usage:
    python src/train_animal_detector.py \
        --data config/animal_species.yaml \
        --epochs 100 \
        --imgsz 640 \
        --model yolov8s.pt

Output:
    runs/detect/animal_species/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="config/animal_species.yaml",
                         help="Path to YOLO dataset yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                         help="Base checkpoint to fine-tune (open-access, free-to-use)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="animal_species")
    parser.add_argument("--patience", type=int, default=20,
                         help="Early-stopping patience (epochs w/o improvement)")
    args = parser.parse_args()

    # yolov8s.pt is loaded from Ultralytics' open-access model zoo on first run.
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        # Augmentations help a lot here since CCTV footage varies in
        # lighting/angle — mosaic + flips give cheap extra robustness.
        mosaic=1.0,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
    )

    # Validate and print metrics — paste these into your README per
    # deliverable #2 (trained weights + performance metrics).
    metrics = model.val()
    print("\n=== VALIDATION METRICS (animal species detector) ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("\nWeights saved to: runs/detect/{}/weights/best.pt".format(args.name))


if __name__ == "__main__":
    main()
