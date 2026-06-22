```markdown
# CityLens Hackathon 2026 — Fallen Tree & Road Obstruction Detection

**Category:** Public Safety Hazards — Collapsed Trees or Structures  
**Dataset:** Fallen Trees (with Palms) — Roboflow Universe  
**Model:** YOLO11n | **mAP@0.5:** 0.799 | **Precision:** 0.832 | **Recall:** 0.714

---

## 1. Approach

This project builds an end-to-end detection and analytics pipeline for identifying
fallen trees and road obstructions from surveillance and roadway imagery, using
YOLO11n as the core detection backbone.

Due to the availability of a single-class training dataset, post-detection analytics
were used to categorize detected obstacles into multiple obstruction severity categories
based on spatial occupancy.

**Pipeline Stages:**

1. Image preprocessing and resize to 640×640
2. YOLO11n inference with confidence threshold = 0.25
3. Bounding box extraction (x, y, width, height per detection)
4. Rule-based obstacle sub-classification using bbox area ratio and aspect ratio
5. Road occupancy % calculation per detection
6. Lane zone assignment (Left / Center / Right) by x-coordinate
7. Severity scoring using occupancy, confidence, and lane weight
8. Export of all detections and analytics to CSV

**Obstacle Sub-Classes (post-detection rule layer):**

| Sub-Class                           | Trigger Condition                  |
|------------------------------------|------------------------------------|
| Large Fallen Tree / Full Road Block | bbox area > 25% of frame          |
| Medium Fallen Tree                  | area 8–25%, aspect ratio ≤ 3.5    |
| Horizontal Log / Branch Debris      | area 8–25%, aspect ratio > 3.5    |
| Small Fallen Tree                   | area 3–8%                         |
| Branch Debris / Minor Obstruction   | area < 3%                         |

**Severity Score Formula:**
```
severity_score = min(road_occupancy% × confidence × lane_weight, 100)
lane_weight = 1.4 for Center Lane, 1.0 otherwise
```

**Severity Levels:**

| Level    | Score Range |
|---------|-------------|
| LOW      | 0 – 10      |
| MEDIUM   | 10 – 30     |
| HIGH     | 30 – 60     |
| CRITICAL | 60 – 100    |

**Dataset:**  
Fallen Trees (with Palms) Object Detection Dataset — Roboflow Universe  
https://universe.roboflow.com/neerumk23-iitk-ac-in/fallen-trees-with-palms-gele3

| Split      | Images |
|-----------|--------|
| Train      | 6,092  |
| Validation | 1,742  |
| Test       | 870    |
| Total      | 8,704  |

---

## 2. Model Architecture

| Parameter          | Value                  |
|-------------------|------------------------|
| Model              | YOLO11n (Ultralytics)  |
| Task               | Object Detection       |
| Number of Classes  | 1                      |
| Class Name         | fallen_tree            |
| Input Resolution   | 640 × 640              |
| Framework          | PyTorch + Ultralytics  |

No architectural modifications were made to the base YOLO11n model.
Standard YOLO11n anchor-free detection head with DFL (Distribution Focal Loss).

**Output per detection:**
- Bounding box: x, y, width, height
- Class label: fallen_tree
- Confidence score: 0.0 – 1.0

---

## 3. Training Details

| Parameter   | Value                                 |
|------------|---------------------------------------|
| Epochs      | 50                                    |
| Batch Size  | 16                                    |
| Image Size  | 640                                   |
| Patience    | 10                                    |
| Optimizer   | AdamW (lr=0.002, Ultralytics default) |
| Hardware    | Kaggle — NVIDIA Tesla T4 GPU          |

**Final Performance Metrics:**

| Metric      | Score |
|------------|-------|
| Precision   | 0.832 |
| Recall      | 0.714 |
| mAP@0.5     | 0.799 |
| mAP@0.5:95  | 0.464 |

Training artifacts (loss curves, confusion matrix, PR curve) are in `training_results/`.

---

## 4. Environment Setup

**Python version:** 3.10+

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
```

Or install directly:
```bash
pip install ultralytics opencv-python pandas numpy torch
```

**Verify installation:**
```python
from ultralytics import YOLO
print("Ultralytics ready")
```

---

## 5. How to Run Inference

**Place input images in a folder, then run:**
```bash
python inference.py --input input_frames --output results
```

**Outputs generated:**
- Annotated images with bounding boxes → output_frames/`
- Detections CSV → predictions.csv

**CSV columns:**
```
frame, class_id, class_name, x_center, y_center, width, height, confidence
```

**Model weights file:** `best.pt`

**Or run directly in Python:**
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("your_image.jpg", conf=0.22)
results[0].show()
```

**Repository Structure:**
```
submission/
├── README.md
├── best.pt
├── inference.py
├── requirements.txt
├── output_frames/
```

---

## 6. Known Issues / Limitations

- **Single class only:** Model is trained on `fallen_tree` exclusively. It cannot
  detect other obstruction types such as fallen poles, barricades, or debris piles.

- **Sub-classification is rule-based:** Obstacle sub-types (Large/Medium/Small tree,
  Branch Debris) are derived from bounding box geometry, not from a trained
  multi-class model. Accuracy of sub-labels depends on image framing and camera angle.

- **Recall gap:** At the default threshold, recall is 0.714 — approximately 1 in 4
  fallen trees in the test set is missed. Lowering `conf` to 0.20 improves recall
  at the cost of more false positives.

- **Domain shift risk:** Model performance may degrade on imagery that differs
  significantly from training data — e.g. nighttime CCTV footage, heavy rain,
  or top-down drone angles.

- **No temporal tracking:** The current pipeline processes individual frames.
  There is no object tracking across video frames, so the same obstruction may
  be counted multiple times in video input.

- **Lane zone logic assumes standard camera angle:** Left/Center/Right lane
  assignment divides the frame into equal thirds. This may be inaccurate for
  angled or fisheye CCTV cameras.

---

## External References

| Resource | Details |
|---------|---------|
| Roboflow Universe | Fallen Trees with Palms Dataset — https://universe.roboflow.com/neerumk23-iitk-ac-in/fallen-trees-with-palms-gele3 |
| Ultralytics Docs | https://docs.ultralytics.com |
| Claude (Anthropic) | Documentation and pipeline structuring |
| ChatGPT | Debugging, implementation support, documentation assistance |
```