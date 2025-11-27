# MedScreenSeeker: Atomic GUI Grounding in Medical Imaging Software

This repository contains the code and a small sample of the **MedScreen** benchmark
used in the project:

> _MedScreenSeeker: Evaluating Vision–Language Models for Atomic GUI Grounding in Medical Imaging Software_

The goal is to evaluate GUI grounding models (e.g., ShowUI, Qwen-based models)
on **atomic click steps** inside four medical imaging tools:
3D Slicer, ITK-SNAP, OHIF viewer, and MicroDICOM.

---

## Repository Structure

```text
.
├── Dataset/               # Sample benchmark data (DOMINO-style format)
│   ├── 3DSlicer/          # <optional> tasks for 3D Slicer
│   ├── ITKSNAP/           # <optional> tasks for ITK-SNAP
│   ├── OHIF/              # sample OHIF tasks (used in examples)
│   └── MicroDICOM/        # <optional> tasks for MicroDICOM
├── Output/
│   └── OHIF/              # Example output folder for predictions & eval
├── run_showui_inference.py    # Run ShowUI on the dataset and save predictions
├── evaluate_gui_grounding.py  # Compute mIoU, Acc@τ and task completion
└── README.md
