"""
src/train_accident_detector.py
Train a YOLOv8 detector that flags accident events in road footage.
This model's detections are the raw input that dark_spot_clustering.py
later aggregates into "dark spot" / "black spot" locations.

Usage:
    python src/train_accident_detector.py \
        --data config/accident_detection.yaml \
        --epochs 100
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="config/accident_detection.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="accident_detector")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        mosaic=1.0,
        fliplr=0.5,
        # Accidents look very different from random angles/lighting on
        # CCTV — heavier color jitter helps generalize across cameras.
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
    )

    metrics = model.val()
    print("\n=== VALIDATION METRICS (accident detector) ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("\nWeights saved to: runs/detect/{}/weights/best.pt".format(args.name))


if __name__ == "__main__":
    main()
