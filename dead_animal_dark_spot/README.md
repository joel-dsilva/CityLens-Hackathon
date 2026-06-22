# CityLens AI Hackathon — KPI Group 4 (Public Safety Hazards)
### Categories: Dark Spots / Black Spots · Dead or Stray Animals on Road

This repo is a starter implementation for the two assigned categories.
It is built to satisfy the hackathon's compulsory deliverables
(category/class label, bounding boxes, bbox coords x/y/w/h per frame)
plus the bonus analytics feature set from the KPI table (severity-style
state classification, count, dwell time, frequency-based dark-spot
identification).

## Architecture

| Category | Approach |
|---|---|
| **Dead/Stray Animals** | Stage 1: YOLOv8 detector → animal species + bbox. Stage 2: YOLOv8-cls on the cropped box → Dead/Alive. Stage 3: ByteTrack (built into Ultralytics) → persistent ID per animal → count + dwell time. |
| **Dark/Black Spots** | YOLOv8 detector flags individual accident events per frame. A separate clustering step (`dark_spot_clustering.py`) aggregates accident detections by location and ranks recurring hotspots by frequency — that ranked list *is* the dark-spot output, not a per-frame detection. |

## Project structure
```
citylens_kpi4/
├── config/
│   ├── animal_species.yaml        # YOLO detection classes (species)
│   ├── dead_alive_classes.yaml    # YOLO classification classes (alive/dead)
│   └── accident_detection.yaml    # YOLO detection classes (accident)
├── dataset_prep/README.md         # how to structure & source your data
├── src/
│   ├── utils.py                   # shared bbox/JSON/CSV export helpers
│   ├── train_animal_detector.py
│   ├── train_dead_alive_classifier.py
│   ├── train_accident_detector.py
│   ├── infer_animal_pipeline.py   # detect + classify + track + dwell time
│   ├── infer_accident_pipeline.py # detect + log accident events
│   └── dark_spot_clustering.py    # frequency-based dark-spot identification
├── outputs/                       # all run outputs land here
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```
All models (YOLOv8 base checkpoints) are open-source/free-to-use via
Ultralytics, satisfying the "open-access and free-to-use" data rule.

## 1. Prepare data
See `dataset_prep/README.md`. If you're starting from the 3 Kaggle
datasets (`picekl/accident`, `bsridevi/modes-dataset-of-stray-animals`,
`siddhi17/road-crossing-dataset`), use the converters in
`dataset_prep/kaggle_converters/` to turn them into this project's
expected folder layout — see that README for which dataset maps to
which category (one of the three doesn't map to either category and
is optional/auxiliary). Update the `path:` field in each
`config/*.yaml` to point at your dataset location once it's ready.

## 2. Train
```bash
# Stage 1 — animal species detector
python src/train_animal_detector.py --data config/animal_species.yaml --epochs 100

# Stage 2 — dead/alive classifier (needs cropped images, see dataset_prep/README.md)
python src/train_dead_alive_classifier.py --data dataset/dead_alive --epochs 50

# Accident detector
python src/train_accident_detector.py --data config/accident_detection.yaml --epochs 100
```
Each script prints final mAP/precision/recall (or top-1/top-5 accuracy)
at the end — paste these into your submission README per deliverable #2.
Target: **>85% accuracy** across assigned categories to be prize-eligible.

## 3. Run inference
```bash
# Animals: detection + dead/alive + count + dwell time
python src/infer_animal_pipeline.py \
    --source path/to/footage.mp4 \
    --species_weights runs/detect/animal_species/weights/best.pt \
    --dead_alive_weights runs/classify/dead_alive_classifier/weights/best.pt \
    --camera_id cam_12 --lat 26.4499 --lon 80.3319

# Accidents: detection + location logging
python src/infer_accident_pipeline.py \
    --source path/to/footage.mp4 \
    --weights runs/detect/accident_detector/weights/best.pt \
    --camera_id cam_12 --lat 26.4499 --lon 80.3319
```
You don't need trained weights to sanity-check the animal pipeline's
plumbing first — run it with `--dry_run` to use stock COCO weights.

This produces, per `outputs/`:
- `bbox_json/<run>/frame_XXXXXX.json` — category + bbox x/y/w/h per frame (deliverable #5)
- `annotated_frames/<run>/frame_XXXXXX.jpg` — output frames with boxes drawn (deliverable #4)
- `<run>_detections.csv` — master log (feeds dwell time + dark-spot clustering)
- `<run>_animal_summary.csv` — per-animal count & dwell time

## 4. Identify dark spots
Run this after accumulating accident logs across multiple cameras/clips
(or point it at historical accident-record CSVs — see `dataset_prep/README.md`):
```bash
python src/dark_spot_clustering.py \
    --logs outputs/*_detections.csv \
    --category accident \
    --mode camera          # or --mode gps --eps_meters 150 --min_samples 3
```
Outputs `outputs/dark_spots_ranked.csv` — locations ranked by accident
frequency, which is the "Identify dark spots (location) by frequency of
accidents" feature from the KPI sheet.

## Notes / assumptions made
- **Why a 2-stage model for animals instead of one combined detector?**
  Keeps each model's label space small and trainable on limited CCTV
  data; the dead/alive classifier can be re-used across all species and
  retrained independently as more data comes in.
- **GPS vs camera-ID dark-spot mode**: use `--mode gps` only if your CCTV
  metadata actually includes coordinates; otherwise `--mode camera`
  (default) ranks dark spots by raw frequency per camera, which still
  satisfies the KPI's "frequency of accidents" requirement without
  needing GPS.
- Swap `yolov8s.pt` / `yolov8s-cls.pt` for `n` (faster, lower accuracy)
  or `m`/`l` (slower, higher accuracy) checkpoints depending on your
  compute budget and the 85% accuracy bar.

## Still to do before submission
- [ ] Source/label training data per `dataset_prep/README.md`
- [ ] Train all 3 models, hit >85% accuracy, record metrics here
- [ ] Run inference over the full provided CCTV test set
- [ ] Document any external datasets/LLMs used (deliverable #6)
- [ ] Package code repo + README + weights + input/output frames + bbox
      coords into a single zip per the submission checklist
