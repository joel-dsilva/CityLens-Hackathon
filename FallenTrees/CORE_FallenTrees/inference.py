import os
import cv2
import csv
import argparse
from ultralytics import YOLO

# ---------------------------------------
# Parse input and output folder arguments
# ---------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Path to input image folder"
)

parser.add_argument(
    "--output",
    required=True,
    help="Path to output folder"
)

args = parser.parse_args()

input_dir = args.input
output_dir = args.output

# ---------------------------------------
# Create output directories
# ---------------------------------------

output_frames_dir = os.path.join(output_dir, "output_frames")
os.makedirs(output_frames_dir, exist_ok=True)

csv_path = "predictions.csv"

# ---------------------------------------
# Load trained model
# ---------------------------------------

model = YOLO("best.pt")

csv_rows = []

# ---------------------------------------
# Run inference on all images
# ---------------------------------------

for image_name in os.listdir(input_dir):

    if not image_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(input_dir, image_name)

    results = model(image_path, conf=0.25)[0]

    annotated = results.plot()

    cv2.imwrite(
        os.path.join(output_frames_dir, image_name),
        annotated
    )

    img_h, img_w = results.orig_shape

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        conf = float(box.conf[0])

        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h

        csv_rows.append({
            "frame": image_name,
            "class_id": 0,
            "class_name": "fallen_tree",
            "x_center": round(x_center, 6),
            "y_center": round(y_center, 6),
            "width": round(width, 6),
            "height": round(height, 6),
            "confidence": round(conf, 6)
        })

# ---------------------------------------
# Save predictions.csv
# ---------------------------------------

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "width",
            "height",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print("Inference completed successfully")
print("Output images saved to:", output_frames_dir)
print("Predictions CSV saved to:", csv_path)