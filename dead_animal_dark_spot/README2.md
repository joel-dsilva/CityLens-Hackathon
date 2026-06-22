# Dataset Preparation Guide

## Your 3 Kaggle datasets — how they map (or don't) to KPI Group 4

| Dataset | Actually is... | Use for |
|---|---|---|
| `picekl/accident` | ACCIDENT benchmark: 2,027 real CCTV accident clips, temporal+spatial+collision-type labels | **Dark Spots** — primary training data for the accident detector |
| `bsridevi/modes-dataset-of-stray-animals` | Raw (unannotated) images of cattle on Indian roads | **Dead/Stray Animals** — Stage 1 species detector only (no bboxes, no dead-animal images — see below) |
| `siddhi17/road-crossing-dataset` | This is actually **INDRA**: pedestrian-POV Indian traffic videos labeled safe/unsafe-to-cross, with vehicle bboxes | **Doesn't fit either category directly.** No accidents, no animals. Optional use: free hard-negative (no-accident) background frames for the accident detector — see `convert_indra_negatives.py`. Skip it if you'd rather keep scope tight. |

I couldn't verify exact Kaggle file layouts first-hand (Kaggle isn't reachable from where these scripts were written), so **always run `kaggle_converters/inspect_dataset.py` on what you actually downloaded before running a converter** — it prints the real folder structure and any CSV headers so you can confirm or adjust column names.

### Step-by-step

```bash
# 0. Download each dataset from Kaggle into its own folder, then inspect it
python dataset_prep/kaggle_converters/inspect_dataset.py /path/to/picekl_accident
python dataset_prep/kaggle_converters/inspect_dataset.py /path/to/modes_cattle
python dataset_prep/kaggle_converters/inspect_dataset.py /path/to/indra            # optional

# 1. Dark Spots: convert ACCIDENT clips into accident-detector training data
python dataset_prep/kaggle_converters/convert_accident_kaggle.py \
    --kaggle_root /path/to/picekl_accident \
    --output_root dataset/accidents

# 1b. (Optional) add INDRA frames as hard negatives to the same folder
python dataset_prep/kaggle_converters/convert_indra_negatives.py \
    --kaggle_root /path/to/indra \
    --output_root dataset/accidents

# 2. Dead/Stray Animals: auto-label MoDES cattle images (Stage 1 only)
python dataset_prep/kaggle_converters/autolabel_cattle_modes.py \
    --images_dir /path/to/modes_cattle/images \
    --output_root dataset/animal_species
```

After this, `dataset/accidents/` and `dataset/animal_species/` are ready
for `train_accident_detector.py` / `train_animal_detector.py` respectively.

**Gaps these 3 datasets don't fill, that you'll still need to source:**
- Dead-animal images (for the Stage 2 dead/alive classifier) — MoDES is all live cattle. Roboflow Universe "roadkill" searches or your own CCTV access are your best bets.
- Non-cattle strays (dogs, goats) for species diversity — MoDES is cattle-only.
- Camera GPS/location metadata for dark-spot clustering — none of these 3 datasets include it; that has to come from the organizers' CCTV camera metadata.

---


## 1. Animal species detector (`dataset/animal_species/`)
Standard YOLO detection layout:
```
dataset/animal_species/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt   # YOLO format: class x_center y_center w h (normalized 0-1)
    labels/val/*.txt
```
Label tool suggestion: [LabelImg](https://github.com/heartexlabs/labelImg) or
[Roboflow](https://roboflow.com) (free tier) exported in "YOLOv8" format.

Good open-access sources to bootstrap from (document these in your README
per deliverable #6 if you use them):
- Roboflow Universe "stray animals" / "street animals" datasets
- Open Images V7 (animal classes, filterable via FiftyOne)
- Your own CCTV dataset access provided by organizers — prioritize this
  since it best matches deployment conditions (camera angle, resolution).

## 2. Dead/Alive classifier (`dataset/dead_alive/`)
ImageNet-style classification layout (see `config/dead_alive_classes.yaml`):
```
dataset/dead_alive/
    train/alive/*.jpg
    train/dead/*.jpg
    val/alive/*.jpg
    val/dead/*.jpg
```
Workflow to build this fast:
1. Run `infer_animal_pipeline.py --dry_run` (or the trained species model)
   on raw footage to get bounding boxes.
2. Crop each detected box out of the frame.
3. Sort crops into `alive/` vs `dead/` folders by eye (dead animals are
   usually lying flat/unnatural posture, no visible motion across frames).
4. Aim for at least 150-200 images per class to start; class balance
   matters more than raw volume for a binary classifier.

## 3. Accident detector (`dataset/accidents/`)
Same YOLO detection layout as animal_species. Useful open-access starting
points:
- Roboflow Universe "Car Accident Detection" / "Accident Detection" datasets
- CADP (Car Accident Detection and Prediction) academic dataset
- Your own CCTV footage, manually labeling collision frames

If your dataset only has accident vs. no-accident at the *clip* level
(not bounding boxes), you can instead start with a classification model
analogous to `train_dead_alive_classifier.py` — swap `--data` to point at
an `accident/` vs `no_accident/` folder structure, and adjust
`infer_accident_pipeline.py` accordingly (use `model.predict` classification
output instead of `.boxes`). Note the hackathon's compulsory deliverables
require bounding boxes, so detection is preferred if you have the labels
for it.

## 4. Dark-spot historical data (optional, no images needed)
If your city already has historical accident *records* (not CCTV footage) —
e.g. a government open-data CSV with date, location, severity — you can feed
those directly into `dark_spot_clustering.py` by reformatting them to match
the expected CSV columns (`category=accident`, `latitude`, `longitude`,
`camera_id`, `timestamp_sec`). This is often higher-quality dark-spot
ground truth than what you'll capture from a 2-week CCTV sample.
