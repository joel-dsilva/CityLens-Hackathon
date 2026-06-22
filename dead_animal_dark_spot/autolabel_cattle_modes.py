"""
dataset_prep/kaggle_converters/autolabel_cattle_modes.py

The Kaggle "bsridevi/modes-dataset-of-stray-animals" (MoDES) dataset is a
curated collection of cattle-on-road images, but it does NOT ship with
bounding-box annotations — it's been used in other papers for tasks
like foreground/background segmentation and depth estimation, which
only need raw images, not boxes. To use it for the animal_species
detector (config/animal_species.yaml), we need bounding boxes.

This script auto-labels the images using a pretrained YOLOv8 model
(stock COCO weights, which already knows the "cow" class — also checks
"horse"/"sheep"/"dog" in case the folder has other strays mixed in) to
generate CANDIDATE bounding boxes. These are a fast starting point, not
ground truth: review them (e.g. in Roboflow or LabelImg) before trusting
them for the >85% accuracy bar, especially for partially-occluded or
multiple-cattle-in-frame images where COCO-pretrained YOLO under-counts.

Usage:
    python dataset_prep/kaggle_converters/autolabel_cattle_modes.py \
        --images_dir /path/to/modes_dataset/images \
        --output_root dataset/animal_species \
        --val_split 0.15

Note: this only handles Stage 1 (species + bbox). MoDES doesn't contain
dead-animal images, so it does NOT help train the dead/alive classifier
(Stage 2) — you'll still need to source/crop dead-animal examples
separately (see dataset_prep/README.md).
"""

import argparse
import os
import random
import shutil

from ultralytics import YOLO

# COCO class indices relevant to roadside stray animals.
# (COCO: 16=bird? no — confirm via model.names at runtime; we filter by name, not index)
RELEVANT_COCO_NAMES = {"cow", "horse", "sheep", "dog", "cat"}

# Map COCO species names -> our project's animal_species.yaml class ids
SPECIES_NAME_MAP = {
    "cow": 1,    # matches config/animal_species.yaml: 1=cow
    "horse": 5,  # 5=horse
    "dog": 0,    # 0=dog
    "cat": 3,    # 3=cat
    "sheep": 7,  # no dedicated sheep class -> bucket into other_animal (7)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True,
                         help="Folder of raw MoDES images (run inspect_dataset.py first "
                              "to confirm the actual subfolder holding the .jpg/.png files)")
    parser.add_argument("--output_root", default="dataset/animal_species")
    parser.add_argument("--model", default="yolov8x.pt",
                         help="Larger model = better candidate boxes for auto-labeling")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    model = YOLO(args.model)
    coco_names = model.names  # {id: name}

    image_files = [
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        raise FileNotFoundError(f"No images found directly in {args.images_dir}. "
                                 "Run inspect_dataset.py to find the right subfolder.")
    print(f"Found {len(image_files)} images in {args.images_dir}")

    random.shuffle(image_files)
    n_val = int(len(image_files) * args.val_split)
    split_for = {f: ("val" if i < n_val else "train") for i, f in enumerate(image_files)}

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_root, "labels", split), exist_ok=True)

    labeled, no_detection = 0, 0
    for fname in image_files:
        split = split_for[fname]
        src_path = os.path.join(args.images_dir, fname)
        dst_img_path = os.path.join(args.output_root, "images", split, fname)
        dst_label_path = os.path.join(args.output_root, "labels", split,
                                       os.path.splitext(fname)[0] + ".txt")

        result = model.predict(src_path, conf=args.conf, verbose=False)[0]
        img_h, img_w = result.orig_shape

        lines = []
        for box, cls_id in zip(result.boxes.xywhn.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            species = coco_names[int(cls_id)]
            if species not in RELEVANT_COCO_NAMES:
                continue
            our_class_id = SPECIES_NAME_MAP.get(species, 7)  # default bucket: other_animal
            xc, yc, w, h = box
            lines.append(f"{our_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            no_detection += 1
            continue  # skip images where the pretrained model found nothing — review these separately

        shutil.copy(src_path, dst_img_path)
        with open(dst_label_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        labeled += 1

    print(f"\nAuto-labeled {labeled} images, {no_detection} had no confident detection "
          f"(left out — review these manually, they may still contain hard-to-spot cattle).")
    print(f"Output -> {args.output_root}/images/{{train,val}}, {args.output_root}/labels/{{train,val}}")
    print("\nReminder: these are MODEL-GENERATED candidate boxes, not human ground truth. "
          "Spot-check a sample (e.g. 50 images) before trusting them for final training, "
          "and consider a quick manual review pass in Roboflow/LabelImg for the worst offenders.")


if __name__ == "__main__":
    main()
