"""
dataset_prep/kaggle_converters/convert_accident_kaggle.py

Converts the Kaggle "picekl/accident" dataset (the ACCIDENT benchmark:
https://www.kaggle.com/datasets/picekl/accident) into the
images/+labels/ layout expected by config/accident_detection.yaml.

Confirmed upstream layout (per the dataset's GitHub companion repo,
accidentbench/ACCIDENT):
    <root>/
        metadata-real.csv
        real_videos/
            videos/*.mp4

IMPORTANT — read before running:
This benchmark provides per-clip annotations for WHEN an accident
happens (temporal) and WHAT type of collision it is (5 classes:
T-bone, Head-on, Rear-end, Sideswipe, Single-vehicle), plus a spatial
annotation *within the frame*. The exact spatial annotation format
(pixel bbox vs. normalized region vs. coarse quadrant) was not
independently verifiable from outside Kaggle, so this script:
  1. Tries several common column-name patterns for time/type/space.
  2. If a usable bbox is found, uses it directly.
  3. If not, falls back to a generous CENTER-WEIGHTED box covering the
     middle ~70% of the frame as a weak label (most CCTV accidents in
     this benchmark are framed center-of-shot). This is a starting
     point, NOT ground truth — spot-check a sample before trusting it
     for the >85% accuracy bar.
  4. ALWAYS prints exactly which columns it matched, so you can verify
     against your actual downloaded metadata-real.csv.

Run dataset_prep/kaggle_converters/inspect_dataset.py on the downloaded
folder first to confirm this layout matches what you actually got.

Usage:
    python dataset_prep/kaggle_converters/convert_accident_kaggle.py \
        --kaggle_root /path/to/picekl_accident \
        --output_root dataset/accidents \
        --frames_per_clip 5 \
        --val_split 0.15
"""

import argparse
import os
import random
import shutil

import cv2
import pandas as pd


# Column-name candidates we'll search for, in priority order.
TIME_COL_CANDIDATES = ["accident_time", "time", "accident_start", "start_time", "timestamp", "frame_time"]
VIDEO_COL_CANDIDATES = ["video", "video_id", "filename", "file", "clip", "clip_id"]
TYPE_COL_CANDIDATES = ["collision_type", "type", "category", "label", "class"]
BBOX_COL_SETS = [
    ["x1", "y1", "x2", "y2"],
    ["x_min", "y_min", "x_max", "y_max"],
    ["bbox_x", "bbox_y", "bbox_w", "bbox_h"],
    ["x", "y", "w", "h"],
]


def find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def find_bbox_columns(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate_set in BBOX_COL_SETS:
        if all(c in cols_lower for c in candidate_set):
            return [cols_lower[c] for c in candidate_set]
    return None


def write_yolo_label(label_path, class_id, x1, y1, x2, y2, img_w, img_h):
    """x1,y1,x2,y2 in pixels -> normalized YOLO xc,yc,w,h"""
    x1, x2 = max(0, x1), min(img_w, x2)
    y1, y2 = max(0, y1), min(img_h, y2)
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    with open(label_path, "w") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def find_video_file(videos_dir, video_ref):
    """video_ref might be a bare id, a filename with extension, or a relative path."""
    candidates = [
        video_ref,
        f"{video_ref}.mp4",
        os.path.basename(str(video_ref)),
        os.path.join(videos_dir, str(video_ref)),
        os.path.join(videos_dir, f"{video_ref}.mp4"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_root", required=True,
                         help="Folder containing metadata-real.csv and real_videos/")
    parser.add_argument("--output_root", default="dataset/accidents")
    parser.add_argument("--frames_per_clip", type=int, default=5,
                         help="How many frames to sample around the labeled accident time")
    parser.add_argument("--window_sec", type=float, default=1.0,
                         help="+/- seconds around accident_time to sample frames from")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    csv_path = os.path.join(args.kaggle_root, "metadata-real.csv")
    if not os.path.isfile(csv_path):
        # fall back: some re-packagings name it labels.csv
        alt = os.path.join(args.kaggle_root, "real_videos", "labels.csv")
        csv_path = alt if os.path.isfile(alt) else csv_path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Couldn't find metadata-real.csv (or real_videos/labels.csv) under {args.kaggle_root}. "
            "Run inspect_dataset.py on this folder and adjust --kaggle_root or this script's csv_path."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with columns: {list(df.columns)}")

    time_col = find_column(df, TIME_COL_CANDIDATES)
    video_col = find_column(df, VIDEO_COL_CANDIDATES)
    type_col = find_column(df, TYPE_COL_CANDIDATES)
    bbox_cols = find_bbox_columns(df)

    print(f"Matched columns -> video: {video_col}, time: {time_col}, "
          f"type: {type_col}, bbox: {bbox_cols}")
    if video_col is None:
        raise ValueError(
            "Could not identify the video-reference column. Open the CSV, find the right "
            "column name, and add it to VIDEO_COL_CANDIDATES at the top of this script."
        )

    videos_dir = os.path.join(args.kaggle_root, "real_videos", "videos")
    if not os.path.isdir(videos_dir):
        videos_dir = os.path.join(args.kaggle_root, "videos")  # fallback layout

    rows = df.to_dict("records")
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_split)
    split_assignment = ["val"] * n_val + ["train"] * (len(rows) - n_val)

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_root, "labels", split), exist_ok=True)

    written, skipped = 0, 0
    for row, split in zip(rows, split_assignment):
        video_ref = row[video_col]
        video_path = find_video_file(videos_dir, video_ref)
        if video_path is None:
            skipped += 1
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine center frame to sample around
        if time_col is not None and pd.notna(row.get(time_col)):
            try:
                center_sec = float(row[time_col])
            except (TypeError, ValueError):
                center_sec = (n_frames / fps) / 2  # fallback: middle of clip
        else:
            center_sec = (n_frames / fps) / 2

        class_name = "accident"
        if type_col is not None and pd.notna(row.get(type_col)):
            # Keep all collision types under one "accident" class for the
            # base detector — swap this to use type_col directly as the
            # class name if you want per-collision-type classes instead.
            class_name = "accident"

        offsets = [0.0] if args.frames_per_clip <= 1 else [
            -args.window_sec + i * (2 * args.window_sec / (args.frames_per_clip - 1))
            for i in range(args.frames_per_clip)
        ]

        base_id = os.path.splitext(os.path.basename(video_path))[0]
        for i, off in enumerate(offsets):
            target_sec = max(0.0, center_sec + off)
            frame_idx = int(target_sec * fps)
            frame_idx = min(frame_idx, max(0, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            img_h, img_w = frame.shape[:2]
            img_name = f"{base_id}_f{frame_idx:06d}.jpg"
            img_path = os.path.join(args.output_root, "images", split, img_name)
            label_path = os.path.join(args.output_root, "labels", split,
                                       img_name.replace(".jpg", ".txt"))
            cv2.imwrite(img_path, frame)

            if bbox_cols:
                x1, y1, x2, y2 = (float(row[c]) for c in bbox_cols)
                # Handle (x,y,w,h) vs (x1,y1,x2,y2) conventions
                if bbox_cols[2].lower() in ("w", "bbox_w"):
                    x2, y2 = x1 + x2, y1 + y2
            else:
                # Weak fallback label: center 70% of the frame.
                x1, y1 = img_w * 0.15, img_h * 0.15
                x2, y2 = img_w * 0.85, img_h * 0.85

            write_yolo_label(label_path, 0, x1, y1, x2, y2, img_w, img_h)
            written += 1

        cap.release()

    print(f"\nDone. Wrote {written} labeled frames, skipped {skipped} rows "
          f"(video file not found — check {videos_dir}).")
    print(f"Output -> {args.output_root}/images/{{train,val}}, {args.output_root}/labels/{{train,val}}")
    if not bbox_cols:
        print("\nNOTE: no bbox columns matched -> used the center-70% weak-label fallback. "
              "Spot-check outputs/accidents samples before training; consider manually "
              "re-annotating a subset with LabelImg/Roboflow if the fallback looks too loose.")


if __name__ == "__main__":
    main()
