Nice, this is a great final step to wrap everything up. I’ll do two things:

1. Propose a **clean repo structure** that matches your two scripts + sample dataset.
2. Give you a **ready-to-paste `README.md`** with clear sections and examples.

---

## 1. Suggested repo structure

You don’t have to follow this exactly, but this layout will make the repo easy to understand:

```text
medscreens eeker/
├── README.md
├── requirements.txt
├── src/
│   ├── run_showui_inference.py      # your first script (slightly path-edited)
│   ├── evaluate_gui_grounding.py    # your second script
│   └── qwen_vl_utils.py             # from the ShowUI/Qwen utils (if needed)
├── data/
│   ├── 3d_slicer/
│   │   ├── Task-001/
│   │   │   ├── annotations.json
│   │   │   ├── 1.png
│   │   │   └── 2.png
│   │   ├── Task-002/
│   │   └── ... (5 tasks total)
│   ├── itk_snap/
│   │   └── Task-XXX/...
│   ├── ohif/
│   │   └── Task-XXX/...
│   └── microdicom/
│       └── Task-XXX/...
└── outputs/
    └── showui_2b/
        └── ohif/
            ├── Task-001/
            │   ├── predictions.jsonl
            │   ├── 1_pred.png
            │   └── 2_pred.png
            ├── Task-002/
            └── _eval_summary.json
```

### Small edits you should make in the scripts

In `run_showui_inference.py` (your first script), set:

```python
# paths relative to the repo root
ROOT_DIR  = "data/ohif"               # or data/3d_slicer, etc.
PRED_ROOT = "outputs/showui_2b/ohif"  # will be created automatically
```

In `evaluate_gui_grounding.py` (your second script), set:

```python
GT_ROOT   = "data/ohif"
PRED_ROOT = "outputs/showui_2b/ohif"
```

Then you can run both scripts from the repo root with:

```bash
python src/run_showui_inference.py
python src/evaluate_gui_grounding.py
```

---

## 2. README.md content

Here is a complete `README.md` you can drop into the repo and tweak project-specific details if you want.

````markdown
# MedScreenSeeker: Atomic GUI Grounding in Medical Imaging Software

This repository contains a **mini-benchmark** and **evaluation code** for atomic GUI grounding in four common medical imaging tools:

- **3D Slicer**
- **ITK-SNAP**
- **OHIF viewer**
- **MicroDICOM**

It accompanies the course project:

> *“MedScreenSeeker: Evaluating Vision–Language Models for Atomic GUI Grounding in Medical Imaging Software”*  
> Mohammad Anas Azeez, MBZUAI, 2025.

The goal is to test how well vision–language models (VLMs) can **click the correct GUI element** given a screenshot and a short instruction.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── run_showui_inference.py      # runs ShowUI-2B on the dataset and saves predictions
│   ├── evaluate_gui_grounding.py    # computes mIoU, Acc@τ, and Task Completion
│   └── qwen_vl_utils.py             # utility from the ShowUI / Qwen-VL codebase
├── data/
│   ├── 3d_slicer/
│   │   └── Task-XXX/ (annotations + screenshots)
│   ├── itk_snap/
│   │   └── Task-XXX/
│   ├── ohif/
│   │   └── Task-XXX/
│   └── microdicom/
│       └── Task-XXX/
└── outputs/
    └── showui_2b/
        └── ohif/
            ├── Task-XXX/
            │   ├── predictions.jsonl
            │   ├── <image>_pred.png
            │   └── eval.json
            └── _eval_summary.json
````

Only a **small subset of tasks** (e.g., 5 per tool) is included here for demonstration.
The full dataset used in the report is kept private due to size and licensing constraints.

---

## Dataset format

Each viewer (e.g., `data/ohif`) contains multiple **task directories**:

```text
data/ohif/
├── Task-001_Load_DICOM/
│   ├── annotations.json
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
├── Task-002_Change_Layout/
│   └── ...
└── ...
```

### `annotations.json`

Each `annotations.json` file stores the **task metadata** and **ground-truth boxes** in a light-weight, DOMINO-style schema:

```json
{
  "task_query": "Load a DICOM study and open it in a 2×2 layout.",
  "images": [
    {
      "file_name": "1.png",
      "width": 2560,
      "height": 1440,
      "instruction": "Click the DICOM button in the top toolbar.",
      "step_index": 1,
      "bboxes": [
        {
          "bbox": [x, y, w, h],    // in pixels, COCO-style XYWH
          "label": "DICOM"
        }
      ]
    },
    {
      "file_name": "2.png",
      "width": 2560,
      "height": 1440,
      "instruction": "Select the study in the DICOM browser.",
      "step_index": 2,
      "bboxes": [
        { "bbox": [x, y, w, h], "label": "Study thumbnail" }
      ]
    }
  ]
}
```

Notes:

* All coordinates in `annotations.json` are **XYWH in pixels**.
* There may be multiple GT boxes per image (e.g., duplicated viewports that are all acceptable clicks).
* Each image corresponds to **one atomic step** in the overall task.

---

## Environment and installation

Create a fresh Python environment (>= 3.10) and install the dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` might look like:

```text
torch>=2.2.0
transformers>=4.43.0
pillow
protobuf
```

You also need:

* A CUDA-capable GPU for running ShowUI-2B (or you can switch to CPU at your own risk).
* Access to the Hugging Face model `showlab/ShowUI-2B` (via `transformers`).

> **Note:** `qwen_vl_utils.py` is copied from the official Qwen-VL / ShowUI repository and slightly adapted for this project.

---

## Running inference (ShowUI-2B example)

The script `src/run_showui_inference.py` runs **ShowUI-2B** over all tasks in a given viewer and saves predictions.

At the top of the file, set:

```python
# Example: run on OHIF sample tasks
ROOT_DIR  = "data/ohif"                   # where annotations.json + images live
PRED_ROOT = "outputs/showui_2b/ohif"      # where predictions and overlays are saved
```

Then from the repo root:

```bash
python src/run_showui_inference.py
```

What the script does:

1. Recursively scans `ROOT_DIR` for task folders with screenshots.
2. For each image:

   * Builds a ShowUI-style prompt that includes the global task query and the step instruction.
   * Calls ShowUI-2B to predict a normalized click point `[x, y]`.
   * Converts the point into a small bounding box in pixel coordinates.
3. Saves:

   * `Task-XXX/predictions.jsonl` — one JSON record per image.
   * `_pred.png` overlay images with red rectangles drawn at the predicted location.
   * A global `all_predictions.jsonl` file in `PRED_ROOT`.

This script is easily adaptable to other VLMs: you mainly need to change the **model loading** and the **output parsing** in `infer_one`.

---

## Running evaluation

The script `src/evaluate_gui_grounding.py` computes:

* Mean IoU (**mIoU**) over all GT boxes.
* Step accuracy at multiple thresholds (e.g., **Acc@0.3**, **Acc@0.5**, **Acc@0.75**).
* Task completion rate, i.e., fraction of tasks where *all* steps reach IoU ≥ τ.

At the top of the file, set:

```python
GT_ROOT   = "data/ohif"                  # same as ROOT_DIR used for inference
PRED_ROOT = "outputs/showui_2b/ohif"     # where predictions.jsonl were written

THRESHOLDS     = [0.3, 0.5, 0.75, 0.9]
PRIMARY_TAU    = 0.3
REQUIRE_LABEL_MATCH = True  # predicted label must match GT instruction string
```

Then run:

```bash
python src/evaluate_gui_grounding.py
```

This will:

1. Find all tasks with `annotations.json` under `GT_ROOT`.
2. For each task, read `Task-XXX/predictions.jsonl` from `PRED_ROOT`.
3. Convert predicted boxes to pixel `xyxy`, compute IoU against GT boxes, and perform greedy matching.
4. Save:

   * `Task-XXX/eval.json` — detailed per-image metrics.
   * `_eval_summary.json` — overall summary + per-task metrics.

The console will also show a short summary:

```text
=== Overall Summary ===
num_tasks: 5
overall_num_gt_boxes: 80
task_completion_rate@0.300: 0.0800
overall_mIoU: 0.2310
Acc@0.3: 0.3660
Acc@0.5: 0.2100
...
```

---

## Extending the repo

You can extend this repository in several ways:

* **Add more viewers or tasks** by dropping additional folders into `data/`.
* **Evaluate other models** (e.g., Qwen2-VL, Qwen3-8B, LLaVA):

  * Implement a new `run_<model>_inference.py` that writes `predictions.jsonl` in the same format.
  * Reuse `evaluate_gui_grounding.py` without modification.
* **Implement MedScreenSeeker**:

  * Add a planner module that performs region proposals.
  * Call the existing grounder on cropped regions and aggregate predictions.

---

## Citation

If you use this repo or the dataset subset in your own work, please cite the project report:

```text
@misc{azeez2025medscreens eeker,
  author       = {Mohammad Anas Azeez},
  title        = {MedScreenSeeker: Evaluating Vision--Language Models for Atomic GUI Grounding in Medical Imaging Software},
  institution  = {MBZUAI},
  year         = {2025}
}
```

---

## Acknowledgements

* This project builds on public implementations of **Qwen-VL** and **ShowUI** for model loading and preprocessing.
* The dataset design and evaluation metrics are inspired by recent GUI grounding work such as ScreenSpot-Pro, OS-Atlas, and SeeClick.

```

You can now:

1. Create a new GitHub repo.  
2. Add your two scripts under `src/`, a minimal subset of tasks under `data/`, and this `README.md`.  
3. Optionally add `requirements.txt` as shown and push.

If you’d like, I can also help you turn the first script into a more generic `--root_dir` / `--pred_root` CLI, but for the assignment this structure + README should already look very clean and professional.
```
