"""
src/infer_accident_pipeline.py
Runs the accident detector over a video / image folder and logs every
accident event with its location (camera_id + lat/lon, or just camera_id
if you only have CCTV-ID-level location data, not GPS).

This log is the raw input to dark_spot_clustering.py, which aggregates
accident frequency by location to surface "dark spots" / "black spots".

Usage:
    python src/infer_accident_pipeline.py \
        --source path/to/video.mp4 \
        --weights runs/detect/accident_detector/weights/best.pt \
        --camera_id cam_12 --lat 26.4499 --lon 80.3319
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(__file__))
from utils import DetectionLogger, draw_box  # noqa: E402

from ultralytics import YOLO


def run(args):
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    run_name = os.path.splitext(os.path.basename(args.source))[0]
    logger = DetectionLogger(args.output_dir, run_name, fps=fps)

    stream = model.predict(source=args.source, stream=True, conf=args.conf, verbose=False)
    accident_frame_count = 0

    for frame_idx, result in enumerate(stream):
        frame = result.orig_img.copy()
        detections_for_log = []

        for box, cls_id, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            category = result.names[int(cls_id)]
            detections_for_log.append(
                {"category": category, "confidence": float(conf), "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
            draw_box(frame, x1, y1, x2, y2, f"{category} {conf:.2f}", color=(0, 0, 255))

        if detections_for_log:
            accident_frame_count += 1

        logger.log_frame(
            frame_idx=frame_idx,
            detections=detections_for_log,
            annotated_frame=frame,
            camera_id=args.camera_id,
            latitude=args.lat,
            longitude=args.lon,
            save_frame=bool(detections_for_log),  # only persist frames with an actual accident
        )

    logger.close()

    print(f"\nFrames with accident detections: {accident_frame_count}")
    print(f"Master detection log (feeds dark-spot clustering) -> {logger.csv_path}")
    print("\nNext step: run src/dark_spot_clustering.py on the accumulated CSV log(s) "
          "across all cameras/runs to identify dark-spot locations by frequency.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="runs/detect/accident_detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--camera_id", default="cam_unknown")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
