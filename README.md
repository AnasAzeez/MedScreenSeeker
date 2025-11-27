Here’s a more detailed `README.md` you can drop straight into your repo (it assumes your current structure with `Dataset/`, `Output/`, `run_showui_inference.py`, and `evaluate_gui_grounding.py`):

---

````markdown
# MedScreenSeeker: Atomic GUI Grounding in Medical Imaging Software

This repository accompanies the project “MedScreenSeeker: Evaluating Vision–Language Models for Atomic GUI Grounding in Medical Imaging Software.”

The repo contains:

- A small sample of the atomic GUI grounding benchmark for medical viewers (subset of the full dataset used in the paper).
- A ShowUI inference script that runs a GUI grounding model on the dataset and saves predictions + visual overlays.
- An evaluation script that computes mIoU, Acc@τ, and Task Completion@τ given ground-truth annotations and predictions.

The code is intentionally lightweight so that you can easily plug in new tools or models.

---

## 1. Repository structure

```text
MedScreenSeeker/
├─ Dataset/                 # Sample annotated tasks (ground truth)
│  └─ OHIF_F/               # Example tool: OHIF viewer
│      ├─ Task-1/
│      │   ├─ annotations.json
│      │   ├─ 1.png
│      │   ├─ 2.png
│      │   └─ ...
│      ├─ Task-2/
│      └─ ...
│
├─ Output/
│  └─ OHIF/                 # Example output folder (predictions + eval)
│      └─ ...               # Created by the scripts
│
├─ run_showui_inference.py  # Runs ShowUI-2B on all tasks and saves predictions
├─ evaluate_gui_grounding.py# Evaluates predictions against ground truth
└─ README.md
````

You can add other viewers (e.g., `3DSlicer`, `ITK_SNAP`, `MicroDICOM`) under `Dataset/` and mirror them under `Output/` if you want to extend the experiments.

---

## 2. Dataset format

Each **tool** (e.g., OHIF) has its own folder in `Dataset/`:

```text
Dataset/OHIF_F/
  Task-1/
    annotations.json
    1.png
    2.png
    ...
  Task-2/
  ...
```

### `annotations.json` schema (per task)

```json
{
  "task_query": "Load a DICOM series and switch to a 2x2 layout",
  "images": [
    {
      "file_name": "1.png",
      "width": 2560,
      "height": 1440,
      "instruction": "Click the DICOM button in the left toolbar",
      "step_index": 1,
      "bboxes": [
        {
          "bbox": [x, y, w, h],
          "label": "click"
        }
      ]
    },
    ...
  ]
}
```

* `bbox` is in **[x, y, width, height]** pixel coordinates.
* Multiple `bboxes` allow multiple valid click locations (e.g., any viewport in a grid layout).

The scripts assume this exact structure for both ground truth and evaluation.

---

## 3. Environment setup

### 3.1. Python and dependencies

Recommended:

* Python **3.10+**
* CUDA-enabled GPU (ShowUI-2B for full-resolution screenshots is heavy; CPU is possible but slow)

Create a new environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA version as needed
pip install transformers pillow google protobuf
pip install git+https://github.com/QwenLM/Qwen2-VL.git  # for qwen_vl_utils
```

> If you run on CPU only, just install the CPU version of PyTorch instead.

---

## 4. Running ShowUI inference

Script: **`run_showui_inference.py`**

This script:

1. Walks through all `Task-*` folders under a dataset root.
2. For each screenshot + instruction:

   * Builds a ShowUI-style prompt.
   * Calls `showlab/ShowUI-2B` (Qwen2-VL–based).
   * Parses the normalized click point `[x, y]`.
   * Converts it to a small pixel bounding box.
3. Saves:

   * A JSONL file with predictions per task.
   * Overlay images with the predicted box drawn on the screenshot.
   * An aggregated `all_predictions.jsonl`.

### 4.1. Configure paths

At the top of `run_showui_inference.py`, set:

```python
ROOT_DIR  = "Dataset/OHIF_F"                 # where Task-* live
PRED_ROOT = "Output/OHIF/Predictions-ShowUI-2B_OHIF"
```

* `ROOT_DIR`: the dataset root for one tool (e.g., `Dataset/OHIF_F`).
* `PRED_ROOT`: where predictions + overlay images will be written.

You can create parallel configs for other tools, e.g.:

```python
ROOT_DIR  = "Dataset/ITK_SNAP"
PRED_ROOT = "Output/ITK/Predictions-ShowUI-2B_ITK"
```

### 4.2. Run the script

```bash
python run_showui_inference.py
```

You should see logs like:

```text
Found 5 task(s).

=== Task-1 :: Load a DICOM series ===
  1.png -> 1_pred.png (boxes: 1)
  2.png -> 2_pred.png (boxes: 1)
  ...
Saved: Output/OHIF/Predictions-ShowUI-2B_OHIF/Task-1/predictions.jsonl
```

Outputs:

* `Output/OHIF/Predictions-ShowUI-2B_OHIF/all_predictions.jsonl`
* `Output/OHIF/Predictions-ShowUI-2B_OHIF/Task-*/predictions.jsonl`
* Overlay PNGs inside each `Task-*` directory.

---

## 5. Evaluating GUI grounding performance

Script: **`evaluate_gui_grounding.py`**

This script compares **predicted boxes** with **ground truth** and computes:

* **mIoU**: mean IoU across all ground-truth boxes.
* **Acc@τ**: fraction of boxes with IoU ≥ τ (for τ ∈ {0.3, 0.5, 0.75, 0.9} by default).
* **TaskCompletion@τ**: fraction of tasks where *all* steps reach IoU ≥ τ.

### 5.1. Configure paths

At the top of `evaluate_gui_grounding.py`, set:

```python
GT_ROOT   = "Dataset/OHIF_F"                        # ground-truth tasks
PRED_ROOT = "Output/OHIF/Predictions-ShowUI-2B_OHIF"  # predictions written by run_showui_inference.py
```

The script expects:

* `GT_ROOT/Task-*/annotations.json`
* `PRED_ROOT/Task-*/predictions.jsonl`

You can change the IoU thresholds if needed:

```python
THRESHOLDS  = [0.3, 0.5, 0.75, 0.9]
PRIMARY_TAU = 0.3         # used for TaskCompletion@τ
REQUIRE_LABEL_MATCH = True  # require predicted label == GT instruction
```

> Setting `REQUIRE_LABEL_MATCH = False` makes evaluation ignore the predicted label and focus only on box geometry.

### 5.2. Run the evaluation

```bash
python evaluate_gui_grounding.py
```

Outputs:

* `Output/OHIF/Predictions-ShowUI-2B_OHIF/_eval_summary.json`
  (overall metrics + per-task breakdown)
* `Output/OHIF/Predictions-ShowUI-2B_OHIF/Task-*/eval.json`
  (detailed per-image IoUs, success flags, etc.)

Console summary (example):

```text
=== Overall Summary ===
num_tasks: 5
overall_num_gt_boxes: 220
task_completion_rate@0.300: 0.0800
overall_mIoU: 0.2145
Acc@0.3: 0.3660
Acc@0.5: 0.1190
Acc@0.75: 0.0150
Acc@0.9: 0.0000

Saved summary: Output/OHIF/Predictions-ShowUI-2B_OHIF/_eval_summary.json
```

---

## 6. Extending to new tools or models

### 6.1. Adding another viewer (e.g., 3D Slicer)

1. Create a new dataset folder under `Dataset/`:

   ```text
   Dataset/3DSlicer_F/
     Task-1/
       annotations.json
       1.png
       ...
   ```

2. Update `ROOT_DIR` and `PRED_ROOT` in `run_showui_inference.py` to point to the new folder.

3. Run inference and evaluation again.

### 6.2. Using a different model

The current script is tailored to **ShowUI-2B** (`showlab/ShowUI-2B`), but it is easy to adapt:

* Replace the model loading block with your model (e.g., a Qwen-VL or OS-Atlas checkpoint).
* Make sure `infer_one(...)` returns:

  * `boxes_px`: list of pixel `[x1, y1, x2, y2]` boxes.
  * `labels`: string labels (usually the instruction text).
  * `prediction`: a dict containing at least `"box_format"` and `"click_boxes"` or a compatible structure.

As long as the predictions JSONL has the same structure, `evaluate_gui_grounding.py` will work unchanged.

---

## 7. Reproducing the paper metrics

The full paper runs several models (Qwen2-7B, Qwen2.5-7B, Qwen3-8B, LLaVA, ShowUI) across **four** medical viewers. This repo provides a **minimal slice** of that benchmark plus the exact evaluation code used for the paper tables and plots.

To approximate the paper setup:

1. Prepare 4 dataset roots under `Dataset/`:

   * `3DSlicer_F/`
   * `ITK_SNAP_F/`
   * `OHIF_F/`
   * `MicroDICOM_F/`
2. For each model and tool:

   * Point `ROOT_DIR` / `PRED_ROOT` in the inference script.
   * Run inference.
   * Run evaluation with the corresponding `GT_ROOT` / `PRED_ROOT`.
3. Aggregate the `_eval_summary.json` files to build tables/plots (e.g., in a separate Jupyter notebook).

---

## 8. Citation

If you use this code or benchmark subset in your work, please cite the project report:

```bibtex
@misc{azeez2025medscreenseeker,
  title        = {MedScreenSeeker: Evaluating Vision--Language Models for Atomic GUI Grounding in Medical Imaging Software},
  author       = {Mohammad Anas Azeez},
  year         = {2025},
  institution  = {MBZUAI},
  note         = {Course project report},
  howpublished = {\url{https://github.com/AnasAzeez/MedScreenSeeker}}
}
```

---

## 9. Contact

For questions or issues, feel free to open a GitHub issue or contact:

**Mohammad Anas Azeez**

📧 `mohammad.azeez@mbzuai.ac.ae`

