# Chest X-Ray Findings Detection via Deep Learning

Senior capstone project (ICT Faculty, Mahidol University) — multi-label classification of chest X-ray findings using an ensemble of CNN architectures, with an emphasis on realistic cross-dataset generalization testing rather than single-dataset validation alone.

## Overview

- Multi-label disease classification on chest X-ray images (NIH ChestX-ray14)
- Ensemble of **DenseNet121**, **ResNet50V2**, and **MobileNetV2**
- Final ensemble performance: **Macro AUC 0.814** (internal validation)
- Trained three view-specific variants — Mixed (AP+PA), AP-only, PA-only — to study how view type affects disease detection
- **Cross-dataset validation on PadChest** (external, unseen hospital/population) to test real-world generalization, not just internal test-set performance
- **Grad-CAM** explainability visualizations for model interpretability
- IEEE-formatted technical report (17 references) covering training strategy, ROC/Youden's Index analysis, and per-disease evaluation

## Key finding

All three model variants showed strong internal validation performance but a **severe performance collapse when evaluated cross-dataset on PadChest** — indicating that domain shift (hospital, equipment, patient population) affects generalization far more than the AP/PA view difference does. This finding, and its implications for deploying chest X-ray models trained on a single institution's data, is analyzed in the full report.

## Engineering notes

- Migrated training infrastructure from an RTX 3050 Ti to an RTX 2080 Ti mid-project, debugging training instability across five training rounds as a result
- Diagnosed a cross-architecture layer-freezing bug: a fixed number of frozen layers worked unevenly across the three backbones. Fixed by switching to **percentage-based unfreezing** (`UNFREEZE_RATIO = 0.28`), which resolved MobileNetV2's underperformance relative to the other two models
- Flagged patient-ID data leakage between train/test splits as a priority fix for result validity, and built a leakage-safe internal test set as a follow-up

## Repository structure

This repository contains the **final (V7) pipeline** and the cross-dataset validation code. Earlier experimental iterations (V2–V6, hybrid variants) and the trained model weights/raw datasets are excluded, to keep the published code focused on the reviewed, final pipeline rather than the full iteration history.

```
├── Calc_Mixed_V1_Per_Model_AUC.py   # Per-model AUC breakdown for the mixed-view ensemble
├── src/
│   ├── Train/                      # Training scripts for the V7 model (Mixed / AP-only / PA-only)
│   ├── Evaluate/                   # Internal + external evaluation, AP vs PA breakdown
│   ├── Predict/                    # Inference / prediction analysis
│   ├── test_cross_dataset/         # PadChest cross-dataset generalization testing
│   ├── create_test_set.py, create_internal_test.py   # Leakage-safe test set construction
│   ├── format_padchest_to_v7*.py, prepare_external_test.py, Expand_Padchest_Test.py  # External dataset preparation
│   ├── Make_NIH_Heatmaps_Complete.py, viewer_v7.py, predict_viewer.py  # Grad-CAM heatmap generation & viewing
│   └── SelectTrainData.py, Train_V7_All_Versions*.py, Evaluate_All_Versions.py, Evaluate_Full_Matrix.py
```

## Tech stack

Python, TensorFlow/Keras, NumPy/Pandas, scikit-learn (ROC/AUC analysis), OpenCV, Grad-CAM
