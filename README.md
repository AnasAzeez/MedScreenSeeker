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
