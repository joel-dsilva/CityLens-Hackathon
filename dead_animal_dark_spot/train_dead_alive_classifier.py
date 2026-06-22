"""
src/train_dead_alive_classifier.py
STAGE 2: Train a YOLOv8 *classification* model on cropped animal images
to predict Dead / Alive (KPI bonus feature set: "Dead or Alive").

Why a separate classifier instead of dead_X / alive_X detection classes?
  - Keeps the species-detector class list small and easy to train with
    limited CCTV data.
  - The crop is already localized to the animal, so a lightweight
    classifier converges fast on a small labeled set.
  - You can re-use the same classifier across all species.

Expected data layout (see config/dead_alive_classes.yaml for details):
    dataset/dead_alive/train/alive/*.jpg
    dataset/dead_alive/train/dead/*.jpg
    dataset/dead_alive/val/alive/*.jpg
    dataset/dead_alive/val/dead/*.jpg

Usage:
    python src/train_dead_alive_classifier.py \
        --data dataset/dead_alive \
        --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/dead_alive",
                         help="Root folder with train/ and val/ subfolders "
                              "(ImageNet-style classification layout)")
    parser.add_argument("--model", default="yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="dead_alive_classifier")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    metrics = model.val()
    print("\n=== VALIDATION METRICS (dead/alive classifier) ===")
    print(f"Top-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")
    print("\nWeights saved to: runs/classify/{}/weights/best.pt".format(args.name))


if __name__ == "__main__":
    main()
