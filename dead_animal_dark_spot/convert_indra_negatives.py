"""
dataset_prep/kaggle_converters/convert_indra_negatives.py

NOTE ON FIT: "siddhi17/road-crossing-dataset" on Kaggle is actually the
INDRA dataset (INdian Dataset for RoAd crossing) — 104 pedestrian-POV
videos of Indian traffic, labeled frame-by-frame as safe/unsafe to
cross, with vehicle bounding boxes. It does NOT contain accident events
or stray-animal dead/alive labels, so it isn't a direct fit for either
"Dark Spots" or "Dead/Stray Animals."

What it IS useful for: free, realistic NEGATIVE (no-accident) frames
for the accident detector. A detector trained only on positive accident
clips tends to false-positive on ordinary heavy traffic; mixing in
clips that are guaranteed accident-free improves precision. This script
extracts a sample of frames from INDRA videos as background/negative
images (empty label files = "no object" for YOLO).

Usage:
    python dataset_prep/kaggle_converters/convert_indra_negatives.py \
        --kaggle_root /path/to/indra_dataset \
        --output_root dataset/accidents \
        --frames_per_video 10

This ADDS negative images into the same dataset/accidents/images/{train,val}
structure used by convert_accident_kaggle.py — run that script first.
"""

import argparse
import os
import random

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_root", required=True,
                         help="Folder containing INDRA's video files "
                              "(run inspect_dataset.py first to find the right subfolder)")
    parser.add_argument("--output_root", default="dataset/accidents")
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    video_files = []
    for dirpath, _, filenames in os.walk(args.kaggle_root):
        for f in filenames:
            if f.lower().endswith((".mp4", ".mov", ".avi")):
                video_files.append(os.path.join(dirpath, f))

    if not video_files:
        raise FileNotFoundError(
            f"No video files found under {args.kaggle_root}. "
            "Run inspect_dataset.py on this folder first to confirm the layout."
        )
    print(f"Found {len(video_files)} video files.")

    random.shuffle(video_files)
    n_val = int(len(video_files) * args.val_split)
    split_for = {v: ("val" if i < n_val else "train") for i, v in enumerate(video_files)}

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_root, "labels", split), exist_ok=True)

    written = 0
    for video_path in video_files:
        split = split_for[video_path]
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            cap.release()
            continue

        sample_idxs = sorted(random.sample(
            range(n_frames), min(args.frames_per_video, n_frames)
        ))
        base_id = os.path.splitext(os.path.basename(video_path))[0]

        for idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            img_name = f"indra_neg_{base_id}_f{idx:06d}.jpg"
            img_path = os.path.join(args.output_root, "images", split, img_name)
            label_path = os.path.join(args.output_root, "labels", split,
                                       img_name.replace(".jpg", ".txt"))
            cv2.imwrite(img_path, frame)
            open(label_path, "w").close()  # empty file = no objects = negative example
            written += 1

        cap.release()

    print(f"\nWrote {written} negative (no-accident) frames into "
          f"{args.output_root}/images/{{train,val}}.")


if __name__ == "__main__":
    main()
