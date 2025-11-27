"""
evaluate_gui_grounding.py

Standalone evaluation script for DOMINO-style UI grounding datasets.

What you need on disk:

GT_ROOT/
  Task-1/
    annotations.json
    1.png
    2.png
    ...
  Task-2/
    annotations.json
    ...
  ...

PRED_ROOT/
  Task-1/
    predictions.jsonl
  Task-2/
    predictions.jsonl
  ...

This script will create, for each task, an `eval.json` file and a global
`_eval_summary.json` under PRED_ROOT.
"""

import os
import json
from typing import List, Dict, Any, Tuple

from PIL import Image


# ---------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------

def xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    area_a  = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b  = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom   = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------
# Loading annotations and predictions
# ---------------------------------------------------------------------

def load_annotations(task_dir: str) -> Dict[str, Any]:
    """
    Load ground-truth boxes from Task-X/annotations.json

    Returns:
      {
        "task_query": str,
        "by_image": {
           "1.png": {
               "boxes": [xyxy...],
               "instruction": str,
               "size": (W,H),
               "step_index": optional
           },
           ...
        }
      }
    """
    annot_path = os.path.join(task_dir, "annotations.json")
    with open(annot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_img = {}
    for im in data.get("images", []):
        W, H = im["width"], im["height"]
        boxes = [xywh_to_xyxy(bb["bbox"]) for bb in im.get("bboxes", [])]
        by_img[im["file_name"]] = {
            "boxes": boxes,
            "instruction": im.get("instruction", ""),
            "size": (W, H),
            "step_index": im.get("step_index"),
        }

    task_query = data.get("task_query") or data.get("task_name") or os.path.basename(task_dir)
    return {"task_query": task_query, "by_image": by_img}


def boxes_px_from_prediction_record(
    rec: Dict[str, Any],
    gt_root: str,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], Tuple[int, int]]:
    """
    Pull predicted boxes (pixel xyxy) + labels from a predictions.jsonl record.

    Prefers `boxes_px` (already pixel coords).
    Otherwise falls back to `prediction.box_format` + `click_boxes`.

    Returns: (boxes_px, labels, image_size)
    """
    # Fast path: use boxes_px if present
    if "boxes_px" in rec and isinstance(rec["boxes_px"], list) and len(rec["boxes_px"]) > 0:
        boxes = [tuple(map(int, b)) for b in rec["boxes_px"]]
        labels = rec.get("labels", [])
        size = tuple(rec.get("image_size", (0, 0)))
        return boxes, labels, size

    # Otherwise, decode from `prediction`
    pred = rec.get("prediction") or rec.get("prediction_raw") or {}
    box_format = (pred.get("box_format") or "").lower()

    size = tuple(rec.get("image_size", (0, 0)))
    if not (size and size[0] > 0 and size[1] > 0):
        rel = rec.get("image_relpath")
        if rel:
            try:
                W, H = Image.open(os.path.join(gt_root, rel)).size
                size = (W, H)
            except Exception:
                pass
    W, H = size if (size and size[0] > 0 and size[1] > 0) else (1, 1)

    boxes_px, labels = [], []
    for item in pred.get("click_boxes", []):
        lab = item.get("label", "box")
        box = item.get("box", [])
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]

        if box_format in ("xyxy_norm", "x1y1x2y2_norm"):
            if max(x1, y1, x2, y2) <= 1.5:
                X1, Y1, X2, Y2 = x1 * W, y1 * H, x2 * W, y2 * H
            else:
                X1, Y1, X2, Y2 = x1, y1, x2, y2
        elif box_format in ("xywh_norm",):
            if max(x1, y1, x2, y2) <= 1.5:
                X1, Y1, X2, Y2 = x1 * W, y1 * H, (x1 + x2) * W, (y1 + y2) * H
            else:
                X1, Y1, X2, Y2 = x1, y1, x1 + x2, y1 + y2
        elif box_format in ("xywh",):
            X1, Y1, X2, Y2 = x1, y1, x1 + x2, y1 + y2
        else:  # xyxy or unknown → assume pixels
            X1, Y1, X2, Y2 = x1, y1, x2, y2

        X1 = max(0, min(int(round(X1)), W - 1))
        Y1 = max(0, min(int(round(Y1)), H - 1))
        X2 = max(0, min(int(round(X2)), W - 1))
        Y2 = max(0, min(int(round(Y2)), H - 1))
        if X2 <= X1:
            X2 = min(W - 1, X1 + 2)
        if Y2 <= Y1:
            Y2 = min(H - 1, Y1 + 2)

        boxes_px.append((X1, Y1, X2, Y2))
        labels.append(lab)

    return boxes_px, labels, (W, H)


def greedy_match_iou(
    gt_boxes: List[Tuple[float, float, float, float]],
    pred_boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, float]]:
    """
    Greedy matching by IoU: returns list of (gt_idx, pred_idx, IoU),
    sorted by IoU desc, no duplicate GT or prediction used.
    """
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            pairs.append((gi, pi, iou_xyxy(gb, pb)))
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_g, used_p, out = set(), set(), []
    for gi, pi, i in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, i))
    return out


# ---------------------------------------------------------------------
# Scoring per task
# ---------------------------------------------------------------------

def score_task(
    task_dir: str,
    pred_task_dir: str,
    primary_tau: float,
    thresholds: List[float],
    require_label_match: bool,
    gt_root: str,
) -> Dict[str, Any]:
    """
    Compare:
      GT:   task_dir/annotations.json
      PRED: pred_task_dir/predictions.jsonl
    """
    gt = load_annotations(task_dir)
    task_query = gt["task_query"]
    gt_by_img = gt["by_image"]

    pred_file = os.path.join(pred_task_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        return {
            "task": os.path.basename(task_dir),
            "task_query": task_query,
            "exists": False,
            "reason": "missing predictions.jsonl",
        }

    preds = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))

    preds_by_img = {p["image"]: p for p in preds}

    per_image_rows = []
    total_gt_boxes = 0
    iou_list_best_per_gt = []
    step_pass = {}

    for fname, meta in gt_by_img.items():
        gt_boxes = meta["boxes"]
        total_gt_boxes += len(gt_boxes)
        instruction = meta.get("instruction", "")
        prec = preds_by_img.get(fname)

        if not prec:
            # no prediction for this image
            ious_for_gts = [0.0] * len(gt_boxes)
            per_image_rows.append({
                "image": fname,
                "instruction": instruction,
                "gt_count": len(gt_boxes),
                "pred_count": 0,
                "best_iou": 0.0,
                "ious_per_gt": ious_for_gts,
                "note": "no_prediction",
            })
            iou_list_best_per_gt.extend(ious_for_gts)
            step_pass[fname] = False
            continue

        pred_boxes, pred_labels, _sz = boxes_px_from_prediction_record(prec, gt_root)

        # Optionally filter predictions by instruction text
        if require_label_match:
            pred_keep = [
                i for i, lab in enumerate(pred_labels)
                if str(lab).strip() == str(instruction).strip()
            ]
            pred_boxes  = [pred_boxes[i] for i in pred_keep]
            pred_labels = [pred_labels[i] for i in pred_keep]

        matches = greedy_match_iou(gt_boxes, pred_boxes)
        matched_iou = [0.0] * len(gt_boxes)
        for gi, pi, iou in matches:
            matched_iou[gi] = max(matched_iou[gi], iou)

        step_success = all(i >= primary_tau for i in matched_iou) if len(gt_boxes) > 0 else True
        step_pass[fname] = step_success

        per_image_rows.append({
            "image": fname,
            "instruction": instruction,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "best_iou": max(matched_iou) if matched_iou else 0.0,
            "ious_per_gt": matched_iou,
            f"step_success@{primary_tau:.3f}": bool(step_success),
        })
        iou_list_best_per_gt.extend(matched_iou if matched_iou else [0.0])

    task_completed = all(step_pass.values()) if len(step_pass) > 0 else False
    mIoU = sum(iou_list_best_per_gt) / max(1, len(iou_list_best_per_gt))
    acc_at = {
        f"Acc@{t:.1f}": sum(1 for v in iou_list_best_per_gt if v >= t) / max(1, len(iou_list_best_per_gt))
        for t in thresholds
    }

    return {
        "task": os.path.basename(task_dir),
        "task_query": task_query,
        "exists": True,
        f"task_completed@{primary_tau:.2f}": bool(task_completed),
        "num_images": len(gt_by_img),
        "num_gt_boxes": int(total_gt_boxes),
        "mIoU": mIoU,
        **acc_at,
        "per_image": per_image_rows,
    }


def eval_list_task_dirs(root_dir: str) -> List[str]:
    """A task dir is any directory in root_dir that contains annotations.json."""
    out = []
    for name in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, name)
        if not os.path.isdir(task_dir):
            continue
        if os.path.exists(os.path.join(task_dir, "annotations.json")):
            out.append(task_dir)
    return out


# ---------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------

def evaluate_all(
    gt_root: str,
    pred_root: str,
    primary_tau: float,
    thresholds: List[float],
    require_label_match: bool,
):
    tasks = eval_list_task_dirs(gt_root)
    print(f"Found {len(tasks)} task(s) with annotations under {gt_root!r}.")

    per_task_results = []
    overall_iou_values = []
    overall_gt_count = 0
    completed = 0

    for task_dir in tasks:
        task_name = os.path.basename(task_dir)
        pred_task_dir = os.path.join(pred_root, task_name)
        res = score_task(
            task_dir,
            pred_task_dir,
            primary_tau=primary_tau,
            thresholds=thresholds,
            require_label_match=require_label_match,
            gt_root=gt_root,
        )
        per_task_results.append(res)

        if res.get("exists"):
            for row in res.get("per_image", []):
                for v in row.get("ious_per_gt", []):
                    overall_iou_values.append(float(v))
                    overall_gt_count += 1
            if res.get(f"task_completed@{primary_tau:.2f}"):
                completed += 1

        out_task_json = os.path.join(pred_root, task_name, "eval.json")
        os.makedirs(os.path.dirname(out_task_json), exist_ok=True)
        with open(out_task_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved per-task eval: {out_task_json}")

    overall_mIoU = (sum(overall_iou_values) / overall_gt_count) if overall_gt_count else 0.0
    overall_acc = {
        f"Acc@{t:.1f}": (
            sum(1 for v in overall_iou_values if v >= t) / overall_gt_count
        ) if overall_gt_count else 0.0
        for t in thresholds
    }
    task_completion_rate = (completed / len(tasks)) if tasks else 0.0

    overall = {
        "num_tasks": len(tasks),
        "overall_num_gt_boxes": int(overall_gt_count),
        f"task_completion_rate@{primary_tau:.3f}": task_completion_rate,
        "overall_mIoU": overall_mIoU,
        **overall_acc,
    }

    out_summary = os.path.join(pred_root, "_eval_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_task": per_task_results}, f, indent=2)

    print("\n=== Overall Summary ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"\nSaved summary: {out_summary}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate GUI grounding predictions against DOMINO-style annotations."
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="data/ohif",
        help="Root directory with GT task folders (default: data/ohif).",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="outputs/showui_2b/ohif",
        help="Root directory with prediction task folders (default: outputs/showui_2b/ohif).",
    )
    parser.add_argument(
        "--primary_tau",
        type=float,
        default=0.3,
        help="Primary IoU threshold for task completion (default: 0.3).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.75, 0.9],
        help="IoU thresholds for Acc@τ metrics (default: 0.3 0.5 0.75 0.9).",
    )
    parser.add_argument(
        "--require_label_match",
        action="store_true",
        help="Require predicted label to match GT instruction exactly.",
    )

    args = parser.parse_args()

    evaluate_all(
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        primary_tau=args.primary_tau,
        thresholds=args.thresholds,
        require_label_match=args.require_label_match,
    )


if __name__ == "__main__":
    main()
