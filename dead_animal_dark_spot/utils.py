"""
src/utils.py
Shared helpers used across the animal-detection and accident-detection
pipelines: drawing annotated frames, and exporting bounding-box results
in the format required by the hackathon deliverables (category, bbox
coords x/y/w/h, clearly labelled, per output frame).
"""

import json
import os
import csv
from datetime import datetime

import cv2
import numpy as np


def xyxy_to_xywh(x1, y1, x2, y2):
    """Convert (x1,y1,x2,y2) corner coords to (x,y,w,h) top-left + size,
    matching the deliverable spec: 'Bounding box dimensions and coords
    (x, y, width, height) clearly labelled'."""
    x = float(x1)
    y = float(y1)
    w = float(x2 - x1)
    h = float(y2 - y1)
    return x, y, w, h


def draw_box(frame, x1, y1, x2, y2, label, color=(0, 0, 255), thickness=2):
    """Draw a single labelled bounding box onto a frame (in-place)."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame,
        (int(x1), int(y1) - text_h - 8),
        (int(x1) + text_w + 4, int(y1)),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (int(x1) + 2, int(y1) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


class DetectionLogger:
    """
    Accumulates per-frame detections and writes them out as:
      - one JSON file per frame  (deliverable #5: bbox coords/dims per frame)
      - one master CSV log across the whole run (used downstream for
        dwell-time calculation and dark-spot clustering)
    """

    def __init__(self, out_dir, run_name, fps=25.0):
        self.out_dir = out_dir
        self.json_dir = os.path.join(out_dir, "bbox_json", run_name)
        self.frames_dir = os.path.join(out_dir, "annotated_frames", run_name)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        self.fps = fps
        self.csv_path = os.path.join(out_dir, f"{run_name}_detections.csv")
        self._csv_initialized = os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not self._csv_initialized:
            self._csv_writer.writerow(
                [
                    "run_name",
                    "frame_idx",
                    "timestamp_sec",
                    "track_id",
                    "category",
                    "confidence",
                    "x",
                    "y",
                    "w",
                    "h",
                    "camera_id",
                    "latitude",
                    "longitude",
                ]
            )

    def log_frame(
        self,
        frame_idx,
        detections,
        annotated_frame=None,
        camera_id="cam_unknown",
        latitude=None,
        longitude=None,
        save_frame=False,
    ):
        """
        detections: list of dicts, each with keys:
            category (str), confidence (float), x1,y1,x2,y2 (px), track_id (optional)
        """
        timestamp_sec = round(frame_idx / self.fps, 3)
        frame_record = {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "camera_id": camera_id,
            "detections": [],
        }

        for det in detections:
            x, y, w, h = xyxy_to_xywh(det["x1"], det["y1"], det["x2"], det["y2"])
            track_id = det.get("track_id", -1)
            frame_record["detections"].append(
                {
                    "category": det["category"],
                    "confidence": round(float(det["confidence"]), 4),
                    "track_id": track_id,
                    "bbox": {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)},
                }
            )
            self._csv_writer.writerow(
                [
                    os.path.basename(self.json_dir),
                    frame_idx,
                    timestamp_sec,
                    track_id,
                    det["category"],
                    round(float(det["confidence"]), 4),
                    round(x, 1),
                    round(y, 1),
                    round(w, 1),
                    round(h, 1),
                    camera_id,
                    latitude,
                    longitude,
                ]
            )

        json_path = os.path.join(self.json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path, "w") as f:
            json.dump(frame_record, f, indent=2)

        if save_frame and annotated_frame is not None:
            frame_path = os.path.join(self.frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, annotated_frame)

        self._csv_file.flush()

    def close(self):
        self._csv_file.close()


def now_iso():
    return datetime.utcnow().isoformat() + "Z"
