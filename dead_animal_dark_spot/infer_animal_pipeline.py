"""
src/infer_animal_pipeline.py
End-to-end inference for "Dead or Stray Animals on Road":
  1. Detect + species-classify animals in each frame (Stage 1 model).
  2. Crop each detection and run the dead/alive classifier (Stage 2 model).
  3. Track animals across frames (ByteTrack via Ultralytics) so the same
     animal keeps one ID across the clip.
  4. Compute COUNT (unique animals) and DWELL TIME (how long each animal
     stays present) per the KPI feature set.
  5. Export annotated frames + per-frame bbox JSON + a summary CSV.

Usage:
    python src/infer_animal_pipeline.py \
        --source path/to/video.mp4 \
        --species_weights runs/detect/animal_species/weights/best.pt \
        --dead_alive_weights runs/classify/dead_alive_classifier/weights/best.pt \
        --camera_id cam_12 --lat 26.4499 --lon 80.3319 \
        --output_dir outputs

If you don't have trained weights yet, you can still test the plumbing
with --dry_run, which uses a stock yolov8n.pt and treats every detection
as species "other_animal" / state "alive" so you can verify the pipeline
end-to-end before your custom models are ready.
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(__file__))
from utils import DetectionLogger, draw_box  # noqa: E402

from ultralytics import YOLO


STATE_COLORS = {"alive": (0, 200, 0), "dead": (0, 0, 200)}


def classify_dead_alive(dead_alive_model, crop):
    """Run the Stage-2 classifier on a single cropped animal image.
    Returns (label:str, confidence:float)."""
    if crop is None or crop.size == 0:
        return "alive", 0.0
    result = dead_alive_model.predict(crop, verbose=False)[0]
    top1 = int(result.probs.top1)
    conf = float(result.probs.top1conf)
    label = result.names[top1]
    return label, conf


def run(args):
    species_model = YOLO(args.species_weights)
    dead_alive_model = None if args.dry_run else YOLO(args.dead_alive_weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    run_name = os.path.splitext(os.path.basename(args.source))[0]
    logger = DetectionLogger(args.output_dir, run_name, fps=fps)

    # track_id -> {first_frame, last_frame, species, states: []}
    track_records = {}

    # model.track() streams frame-by-frame and assigns persistent IDs
    # via ByteTrack, so the same animal keeps one ID across the video.
    stream = species_model.track(
        source=args.source,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=args.conf,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        frame = result.orig_img.copy()
        detections_for_log = []

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                track_id = int(track_id)
                species = result.names[int(cls_id)]

                # Stage 2: crop + classify dead/alive
                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if args.dry_run:
                    state, state_conf = "alive", 1.0
                else:
                    state, state_conf = classify_dead_alive(dead_alive_model, crop)

                # Update per-track dwell-time bookkeeping
                rec = track_records.setdefault(
                    track_id, {"first_frame": frame_idx, "last_frame": frame_idx,
                               "species": species, "states": []}
                )
                rec["last_frame"] = frame_idx
                rec["states"].append(state)

                category = f"{species}_{state}"
                detections_for_log.append(
                    {
                        "category": category,
                        "confidence": float(conf),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "track_id": track_id,
                    }
                )

                color = STATE_COLORS.get(state, (0, 165, 255))
                label = f"#{track_id} {species} ({state} {state_conf:.2f})"
                draw_box(frame, x1, y1, x2, y2, label, color=color)

        logger.log_frame(
            frame_idx=frame_idx,
            detections=detections_for_log,
            annotated_frame=frame,
            camera_id=args.camera_id,
            latitude=args.lat,
            longitude=args.lon,
            save_frame=(frame_idx % args.save_every == 0),
        )

        if args.show:
            cv2.imshow("CityLens - Animal Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    logger.close()
    cv2.destroyAllWindows()

    # ---- Summary: count + dwell time per animal ----
    summary_path = os.path.join(args.output_dir, f"{run_name}_animal_summary.csv")
    with open(summary_path, "w") as f:
        f.write("track_id,species,majority_state,dwell_time_sec,first_seen_frame,last_seen_frame\n")
        for tid, rec in track_records.items():
            dwell_sec = (rec["last_frame"] - rec["first_frame"] + 1) / fps
            majority_state = max(set(rec["states"]), key=rec["states"].count)
            f.write(
                f"{tid},{rec['species']},{majority_state},{dwell_sec:.2f},"
                f"{rec['first_frame']},{rec['last_frame']}\n"
            )

    print(f"\nUnique animals detected: {len(track_records)}")
    print(f"Per-animal count & dwell-time summary -> {summary_path}")
    print(f"Per-frame bbox JSON -> {logger.json_dir}")
    print(f"Annotated frames -> {logger.frames_dir}")
    print(f"Master detection log (CSV, feeds dark-spot/animal analytics) -> {logger.csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to video file (or webcam index)")
    parser.add_argument("--species_weights", default="runs/detect/animal_species/weights/best.pt")
    parser.add_argument("--dead_alive_weights", default="runs/classify/dead_alive_classifier/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--camera_id", default="cam_unknown")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_every", type=int, default=1, help="Save annotated frame every N frames")
    parser.add_argument("--show", action="store_true", help="Show a live preview window")
    parser.add_argument("--dry_run", action="store_true",
                         help="Skip dead/alive classifier (pipeline smoke test)")
    args = parser.parse_args()

    if args.dry_run:
        args.species_weights = "yolov8n.pt"  # generic COCO weights, just to test plumbing

    os.makedirs(args.output_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
