"""
run_showui_inference.py

Run ShowUI-2B on a DOMINO-style GUI grounding dataset and save predictions.

Expected directory layout (example for OHIF):

data/
  ohif/
    Task-001_Load_DICOM/
      annotations.json
      1.png
      2.png
      ...
    Task-002_Change_Layout/
      ...

Outputs will be written under e.g.:

outputs/
  showui_2b/
    ohif/
      Task-001_Load_DICOM/
        predictions.jsonl
        1_pred.png
        2_pred.png
        ...
      _eval_summary.json  (created by the separate evaluation script)
"""

import os
import re
import json
import glob
import ast
from typing import List, Dict, Any, Tuple

import torch
import transformers
import google.protobuf  # noqa: F401
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info  # must be in the same folder or on PYTHONPATH


# ---------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------

FILE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def to_file_url(p: str) -> str:
    """Turn a local path into a file:// URL for the vision model."""
    if p.startswith(("http://", "https://", "file://")):
        return p
    ap = os.path.abspath(p)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"Missing image: {p}")
    return "file://" + ap.replace("\\", "/")


def natural_key(s: str):
    """Sort filenames with embedded numbers in human order (1,2,10)."""
    base = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


# ---------------------------------------------------------------------
# Drawing and boxes (pixels)
# ---------------------------------------------------------------------

def draw_and_save(
    image_path: str,
    boxes_xyxy_px: List[Tuple[int, int, int, int]],
    labels: List[str],
    out_path: str,
    outline_width: int = 5,
):
    """
    Draw solid red rectangles + label chips on an image and save to disk.

    All coordinates are in *pixels*.
    """
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    labels = labels or [f"box{i+1}" for i in range(len(boxes_xyxy_px))]

    for (x1, y1, x2, y2), lab in zip(boxes_xyxy_px, labels):
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=outline_width)

        # Draw a simple label chip just above the box
        try:
            l, t, r, b = d.textbbox((0, 0), lab)
            tw, th = r - l, b - t
        except Exception:
            tw, th = max(8 * len(lab), 16), 16

        pad = 4
        chip = [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1]
        d.rectangle(chip, fill=(255, 0, 0))
        d.text((chip[0] + pad, chip[1] + pad), lab, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=95)


def point_to_box(
    x_norm: float,
    y_norm: float,
    image_path: str,
    box_frac: float = 0.05,
) -> Tuple[int, int, int, int]:
    """
    Convert a normalized point [0,1]×[0,1] into a small bounding box around it.
    Output box is in *pixel* coordinates (x1, y1, x2, y2).
    """
    img = Image.open(image_path)
    W, H = img.size

    x_norm = float(max(0.0, min(1.0, x_norm)))
    y_norm = float(max(0.0, min(1.0, y_norm)))

    cx = x_norm * W
    cy = y_norm * H
    box_size = max(8, int(min(W, H) * box_frac))

    x1 = int(round(cx - box_size / 2))
    y1 = int(round(cy - box_size / 2))
    x2 = int(round(cx + box_size / 2))
    y2 = int(round(cy + box_size / 2))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 2)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 2)
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------
# Task metadata and prompts
# ---------------------------------------------------------------------

def load_task_meta(task_dir: str):
    """
    Load task-level metadata.

    Returns:
      task_query: str
      instr_by_file: dict like {"1.png": "Open the View tab", ...}
    """
    annot = os.path.join(task_dir, "annotations.json")
    task_query = os.path.basename(task_dir)
    instr_by_file: Dict[str, str] = {}

    if os.path.exists(annot):
        with open(annot, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_query = data.get("task_query") or data.get("task_name") or task_query
        for im in data.get("images", []):
            fname = im.get("file_name")
            instr = im.get("instruction") or ""
            if fname:
                instr_by_file[fname] = instr
    return task_query, instr_by_file


# ShowUI-specific image size constraints (from README)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28

SYSTEM_PROMPT = (
    "You are ShowUI, a vision-language-action model for GUI agents.\n"
    "Given a screenshot and a natural language description of the next UI step, "
    "predict the single best clickable point as a Python list [x, y], where x and y "
    "are floats in [0, 1] (relative coordinates on the screenshot).\n"
    "Respond with ONLY the list, for example: [0.25, 0.73]."
)


def build_messages_for_image(
    image_path: str,
    task_query: str,
    image_instruction: str,
) -> List[Dict[str, Any]]:
    """
    Build ShowUI-style chat messages:
      role='user' with:
        - system-like text
        - image
        - query text
    """
    if image_instruction:
        query = f"Global task: {task_query}\nStep instruction: {image_instruction}"
    else:
        query = f"Global task: {task_query}\nReturn the clickable point for this screen."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {
                    "type": "image",
                    "image": to_file_url(image_path),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": query},
            ],
        }
    ]
    return messages


# ---------------------------------------------------------------------
# ShowUI-2B model setup
# ---------------------------------------------------------------------

def load_model_and_processor(model_id: str, device: str | None = None):
    """Load ShowUI-2B (or compatible) model and processor."""
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("protobuf:", google.protobuf.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Avoid tie_weights error by disabling word embedding tying
    config.tie_word_embeddings = False

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )

    # Enforce deterministic generation (no sampling)
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.temperature = 0.0
    gen_cfg.top_k = 1
    gen_cfg.top_p = 1.0

    print("Model loaded:", model_id)
    print("Running on device:", device)
    return model, processor, device


# ---------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------

@torch.inference_mode()
def infer_one(
    model,
    processor,
    image_path: str,
    task_query: str,
    image_instruction: str,
    device: str,
    max_new_tokens: int = 64,
):
    messages = build_messages_for_image(image_path, task_query, image_instruction)

    # 1) Build chat template text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2) Vision inputs (images/videos)
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) Tokenize + process
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # 4) Generate
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        top_k=None,
    )

    # 5) Decode only new tokens
    new_tokens = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    txt = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # 6) Parse output as [x, y] (normalized)
    coords = None
    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            coords = (float(parsed[0]), float(parsed[1]))
        elif (
            isinstance(parsed, (list, tuple))
            and len(parsed) > 0
            and isinstance(parsed[0], (list, tuple))
            and len(parsed[0]) >= 2
        ):
            coords = (float(parsed[0][0]), float(parsed[0][1]))
    except Exception as e:
        print(f"[ERROR] ast.literal_eval failed for {os.path.basename(image_path)}: {e}")
        print("RAW[:200]:", txt[:200])

    boxes_px: List[Tuple[int, int, int, int]] = []
    click_boxes: List[Dict[str, Any]] = []

    if coords is not None:
        x_norm, y_norm = coords
        box = point_to_box(x_norm, y_norm, image_path)
        boxes_px.append(box)

        click_boxes.append(
            {
                "label": image_instruction or "",
                "box": [box[0], box[1], box[2], box[3]],  # pixel xyxy
            }
        )

    labels = [image_instruction or "click"] * len(boxes_px)
    W, H = Image.open(image_path).size

    prediction = {
        "image": os.path.basename(image_path),
        "box_format": "xyxy",   # pixel coordinates
        "click_boxes": click_boxes,
        "raw_text": txt,
    }

    return {
        "raw_text": txt,
        "prediction_raw": prediction,
        "prediction": prediction,
        "label_overridden": False,
        "boxes_px": boxes_px,
        "labels": labels,
        "image_size": (W, H),
    }


# ---------------------------------------------------------------------
# Dataset traversal helpers
# ---------------------------------------------------------------------

def is_task_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    basename = os.path.basename(path)
    if basename.lower().startswith("predictions-"):
        return False
    for ext in FILE_EXTS:
        if glob.glob(os.path.join(path, f"*{ext}")):
            return True
    return False


def list_task_dirs(root_dir: str) -> List[str]:
    return [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir), key=natural_key)
        if is_task_dir(os.path.join(root_dir, name))
    ]


def run_all_tasks(root_dir: str, pred_root: str, model_id: str = "showlab/ShowUI-2B"):
    """
    Run ShowUI inference on every task directory under root_dir and
    write predictions + overlay images under pred_root.
    """
    os.makedirs(pred_root, exist_ok=True)

    model, processor, device = load_model_and_processor(model_id)
    task_dirs = list_task_dirs(root_dir)
    print(f"Found {len(task_dirs)} task(s) under {root_dir!r}.")

    agg_path = os.path.join(pred_root, "all_predictions.jsonl")
    with open(agg_path, "w", encoding="utf-8") as _:
        pass  # truncate

    for task_dir in task_dirs:
        task_label = os.path.basename(task_dir)
        task_query, instr_by_file = load_task_meta(task_dir)

        out_task_dir = os.path.join(pred_root, task_label)
        os.makedirs(out_task_dir, exist_ok=True)
        pred_jsonl_path = os.path.join(out_task_dir, "predictions.jsonl")

        image_paths: List[str] = []
        for ext in FILE_EXTS:
            image_paths += glob.glob(os.path.join(task_dir, f"*{ext}"))
        image_paths = sorted(set(image_paths), key=natural_key)

        print(f"\n=== {task_label} :: {task_query} ===")

        with open(pred_jsonl_path, "w", encoding="utf-8") as f_out, \
             open(agg_path, "a", encoding="utf-8") as agg_out:

            for img_path in image_paths:
                fname = os.path.basename(img_path)
                instruction = instr_by_file.get(fname, "")

                res = infer_one(
                    model,
                    processor,
                    image_path=img_path,
                    task_query=task_query,
                    image_instruction=instruction,
                    device=device,
                    max_new_tokens=64,
                )

                boxes_px = res["boxes_px"]
                labels = res["labels"]

                record = {
                    "task": task_label,
                    "task_query": task_query,
                    "image": fname,
                    "instruction": instruction,
                    "image_relpath": os.path.relpath(img_path, root_dir),
                    "image_size": res["image_size"],
                    "prediction": res["prediction"],  # includes pixel xyxy boxes
                    "boxes_px": boxes_px,
                    "labels": labels,
                    "label_overridden": res["label_overridden"],
                }

                line = json.dumps(record, ensure_ascii=False)
                f_out.write(line + "\n")
                agg_out.write(line + "\n")

                overlay_path = os.path.join(
                    out_task_dir, f"{os.path.splitext(fname)[0]}_pred.png"
                )
                draw_and_save(img_path, boxes_px, labels, overlay_path)

                print(
                    f"  {fname} -> {os.path.relpath(overlay_path, out_task_dir)} "
                    f"(boxes: {len(boxes_px)})"
                )

        print(f"Saved predictions: {pred_jsonl_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ShowUI-2B on a GUI grounding dataset and save predictions."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/ohif",
        help="Root directory with task folders (default: data/ohif).",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="outputs/showui_2b/ohif",
        help="Output directory for predictions (default: outputs/showui_2b/ohif).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="showlab/ShowUI-2B",
        help="Hugging Face model ID for ShowUI (default: showlab/ShowUI-2B).",
    )

    args = parser.parse_args()
    run_all_tasks(args.root_dir, args.pred_root, model_id=args.model_id)


if __name__ == "__main__":
    main()
