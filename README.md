# Adaptive Cross-Modal Fusion with Sparse Attention for Pedestrian Crossing Intention Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 1. Overview

This repository contains the official implementation of:

> **“Adaptive Cross-Modal Fusion with Sparse Attention for Pedestrian Crossing Intention Prediction”**  
> *Md Mahfuzur Rahman, Pengzhan Zhou, A. F. M. Abdun Noor, Md Imam Ahasan, Kah Ong Michael Goh, S. M. Hasan Mahmud, Md Mustafizur Rahman, Kaixin Gao*  
> Submitted to **PeerJ Computer Science (Applications of  AI category)**, April 2026.

---

## 2. Project Description

This repository provides the official implementation of **ADAPT (Adaptive Domain-Aware Pedestrian Crossing Transformer)**, a multimodal deep learning framework for pedestrian crossing intention prediction in autonomous driving scenarios. ADAPT is designed to model complex interactions between visual context and motion dynamics by jointly processing multiple complementary modalities over a short temporal window. The model takes as input a sequence of 16 consecutive frames and integrates four spatially aligned visual modalities—RGB images, local depth maps, global semantic segmentation maps, and global depth maps—along with three kinematic signals: ego-vehicle speed, pedestrian bounding box trajectories, and skeleton pose descriptors.

The architecture consists of five main components:
- A **Visual Feature Encoding (VFE)** module based on a weight-shared Swin Transformer V2 backbone for extracting spatial representations from image-based modalities.
- A **Cross-Modality Guided Attention (CMGA)** module for fusing local and global visual features through hierarchical channel-spatial attention with adaptive routing.
- A **Motion Feature Encoding (MFE)** module using a Mamba State Space Model to capture temporal dependencies in kinematic signals.
- A **Sparse Cross-Modal Attention (SCMA)** module that selectively integrates multimodal representations by retaining only the most informative inter-modal interactions.
- A **Temporal Feature Fusion (TFF)** module that aggregates frame-level features and predicts pedestrian crossing intention.

The proposed approach is evaluated on two publicly available benchmarks, JAAD and PIE, where it achieves state-of-the-art performance while maintaining low inference latency. The framework is designed with a focus on both predictive accuracy and computational efficiency, making it suitable for real-time deployment in intelligent transportation systems. This repository includes the model implementation, training and evaluation pipelines, and configuration files required to reproduce the reported results.

---

## 3. Key Features

- **Multimodal Input Integration:** Processes four complementary visual modalities (RGB, local depth, global semantic segmentation, and global depth) together with kinematic signals (ego-vehicle speed, pedestrian bounding boxes, and pose descriptors) for robust intention prediction.

- **Adaptive Cross-Modal Fusion (CMGA):** Introduces a hierarchical channel-spatial attention mechanism with a learnable routing gate to dynamically fuse local and global visual context based on scene conditions.

- **Sparse Cross-Modal Attention (SCMA):** Employs a top-k sparsity constraint to retain only the most informative inter-modal interactions, reducing noise from irrelevant modality pairings and improving efficiency.

- **Temporal Motion Modeling with Mamba (MFE):** Utilizes a Mamba State Space Model to capture long-range temporal dependencies in kinematic signals with linear computational complexity.

- **Transformer-based Temporal Aggregation (TFF):** Aggregates frame-level multimodal features using a Vision Transformer-style encoder for accurate sequence-level prediction.

- **Efficiency–Accuracy Trade-off:** Achieves strong predictive performance on JAAD and PIE benchmarks while maintaining low inference latency suitable for real-time applications.

- **Modular and Extensible Design:** The architecture is organized into independent modules, enabling easy modification, ablation, and extension for future research.

- **Reproducible Training Pipeline:** Includes configuration files and scripts for training, evaluation, and benchmarking under standardized experimental settings.

---

## 4. Datasets

This work is evaluated on two publicly available benchmarks for pedestrian crossing intention prediction: **JAAD** and **PIE**. These datasets provide annotated urban driving scenarios with pedestrian behavior labels and are widely used in the literature. The datasets are **not included in this repository** and must be downloaded from their official sources.

### 4.1 JAAD (Joint Attention in Autonomous Driving)

- **DOI:** https://doi.org/10.48550/arXiv.1609.04741  
- **Website:** https://data.nvision2.eecs.yorku.ca/JAAD_dataset/  

The JAAD dataset consists of 346 video sequences captured from a forward-facing camera mounted on a vehicle in urban environments. It includes detailed annotations such as pedestrian bounding boxes, crossing behavior labels, occlusion levels, and contextual attributes (e.g., traffic signals and scene conditions).

**Dataset Characteristics:**
- Monocular urban driving videos
- Annotated pedestrian behaviors (crossing / not crossing)
- Bounding boxes and contextual attributes
- Designed for studying human–vehicle interaction

**Experimental Protocol (used in this work):**
- Training: 177 videos  
- Validation: 29 videos  
- Test: 117 videos  
- Two evaluation subsets:
  - **JAADbeh:** pedestrians with behavior annotations  
  - **JAADall:** all annotated pedestrians  

**Clip Extraction:**
- Observation window: 16 frames  
- Temporal overlap: 80% (stride = 3 frames)  
- Time-to-event (TTE): 1–2 seconds  

**Preprocessing Summary:**
- Frames resized to 256 × 256  
- RGB images normalized using ImageNet statistics  
- Bounding boxes normalized by image dimensions  
- Ego-vehicle speed normalized to [0, 1] using 60 km/h as maximum  

---

### 4.2 PIE (Pedestrian Intention Estimation)

- **DOI:** https://doi.org/10.1109/ICCV.2019.00636  
- **Website:** https://data.nvision2.eecs.yorku.ca/PIE_dataset/  

The PIE dataset contains continuous driving sequences collected under diverse environmental conditions, including varying illumination, traffic density, and pedestrian behaviors. It provides richer ego-motion signals compared to JAAD, making it suitable for studying temporal dynamics in pedestrian intention prediction.

**Dataset Characteristics:**
- Continuous driving sequences across multiple sessions
- Rich ego-vehicle telemetry (speed and motion)
- Diverse environmental and behavioral conditions

**Experimental Protocol (used in this work):**
- Training: set01, set02, set06  
- Validation: set04, set05  
- Test: set03  

**Clip Extraction:**
- Observation window: 16 frames  
- Temporal overlap: 80%  
- Time-to-event (TTE): 1–2 seconds  

**Preprocessing Summary:**
- Frames resized to 256 × 256  
- RGB and semantic inputs normalized using ImageNet statistics  
- Depth maps replicated to 3 channels and normalized  
- Bounding boxes normalized by image size  
- Ego-vehicle speed normalized to [0, 1] using 70 km/h as maximum  

---

### 4.3 Important Notes

- The JAAD and PIE datasets are **third-party datasets** and must be obtained from their respective official sources.
- Users are responsible for complying with the datasets’ **licenses and usage policies**.
- This repository assumes that the datasets are organized and preprocessed into modality-aligned clips before training (see *Methodology* and *Usage Instructions* sections for details).

---

## 5. Code Information
The repository is designed with a modular, scalable, and research-oriented structure, enabling easy experimentation, reproducibility, and extension.
**Repository layout :**
```bash
adapt_repo/
├── adapt/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Multimodal clip dataset and sample parsing
│   │   ├── manifest.py             # Manifest loading utilities
│   │   ├── samplers.py             # Weighted sampling helpers
│   │   └── transforms.py           # Synchronized multimodal preprocessing/augmentation
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── evaluator.py            # Validation/test evaluation and metric computation
│   │   ├── infer.py                # Inference pipeline
│   │   └── trainer.py              # Training loop, AMP, DDP, checkpointing
│   ├── losses/
│   │   ├── __init__.py
│   │   └── intent_loss.py          # Composite loss: weighted BCE + head regularization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── adapt.py                # Full ADAPT model assembly
│   │   ├── cmga.py                 # Cross-Modality Guided Attention module
│   │   ├── mfe.py                  # Motion Feature Encoding with Mamba SSM
│   │   ├── scma.py                 # Sparse Cross-Modal Attention
│   │   ├── swin_vfe.py             # Visual Feature Encoding with Swin Transformer V2 Tiny
│   │   └── tff.py                  # Temporal Feature Fusion transformer
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpoint.py           # Save/load checkpoints
│   │   ├── config.py               # YAML config parsing
│   │   ├── distributed.py          # DDP setup/cleanup helpers
│   │   ├── logger.py               # Console/file/TensorBoard logging
│   │   ├── metrics.py              # Acc, AUC, F1, Precision, Recall
│   │   ├── reproducibility.py      # Seeds and deterministic setup
│   │   └── schedulers.py           # Stage-wise LR and freeze/unfreeze control
│   └── version.py
├── configs/
│   ├── base.yaml                   # Shared default configuration
│   ├── jaad_all.yaml               # JAAD_all experiment config
│   ├── jaad_beh.yaml               # JAAD_beh experiment config
│   └── pie.yaml                    # PIE experiment config
├── scripts/
│   ├── train.py                    # Main training entrypoint
│   ├── eval.py                     # Evaluation entrypoint
│   └── infer.py                    # Inference entrypoint
├── tests/
│   ├── test_cmga.py
│   ├── test_mfe.py
│   ├── test_scma.py
│   ├── test_tff.py
│   └── test_model_shapes.py
├── assets/
│   └── architecture.txt            # Text-based architecture diagram / notes
├── PAPER_BREAKDOWN.md              # Structured technical breakdown of the paper
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation script
└── .gitignore
```

### Code Usage Summary

The repository provides a complete pipeline for training, evaluation, and inference of the ADAPT model. The workflow is configuration-driven and supports reproducible experiments across JAAD and PIE datasets.

#### 1. Training

To train the model on a specific dataset, use:

```bash
python scripts/train.py --config configs/jaad_beh.yaml
```
#### 2. Evaluation
To evaluate a trained model:
```bash
python scripts/eval.py --config configs/jaad_beh.yaml --checkpoint path/to/checkpoint.pth
```

---

## 6. Method summary
a multimodal architecture designed to predict whether a pedestrian will cross the road within a short future horizon (1–2 seconds). The method combines visual context understanding and temporal kinematic modeling, with a focus on adaptive and sparse cross-modal fusion.

**Overall Pipeline**
Given a 16-frame clip, the model processes multimodal inputs through five sequential modules:
```bash
Visual Inputs (4 modalities) ──► VFE ──► CMGA ──┐
                                               ├──► SCMA ──► TFF ──► Prediction
Kinematic Inputs (speed, bbox, pose) ─► MFE ───┘
```

**Training Objective**
The model is trained using a composite loss:
- Class-weighted Binary Cross-Entropy
- L2 regularization on classifier weights
```bash
L = L_intent + μ · L_reg   (μ = 1e-3)
```
**Key Design Principles**
- Adaptive fusion: CMGA dynamically selects useful visual context
- Selective interaction: SCMA avoids over-attending across modalities
- Efficiency: Mamba replaces quadratic attention for temporal modeling
- Hierarchical reasoning: Separates local (pedestrian-level) and global (scene-level) cues
  
---
## 7. Installation
This repository is implemented in PyTorch and supports distributed training (DDP) and mixed precision (AMP) as described in the paper

**Requirements**
- Python ≥ 3.9
- CUDA ≥ 11.8 
- PyTorch ≥ 2.1
- timm ≥ 0.9.12
- mamba-ssm ≥ 1.2.0

```bash
conda create -n adapt python=3.10 -y
conda activate adapt
pip install -e .
```

Tested dependency targets from the paper:

- `torch >= 2.1.0`
- `timm >= 0.9.12`
- `mamba-ssm >= 1.2.0` fileciteturn2file0L311-L325

---

## 8. Dataset preparation

The repository uses **manifest files** for reproducibility. Each JSON manifest is a list of clip records. Every record must contain:

```json
{
  "sample_id": "unique_clip_id",
  "dataset": "jaad_beh",
  "label": 1,
  "tte": 1.4,
  "rgb_paths": ["...16 paths..."],
  "local_depth_paths": ["...16 paths..."],
  "global_semantic_paths": ["...16 paths..."],
  "global_depth_paths": ["...16 paths..."],
  "speed": [16 scalars],
  "bbox": [[x1,y1,x2,y2], "...16 rows..."],
  "pose": [[36 values], "...16 rows..."]
}
```

### 9. Modality Preparation Notes

The ADAPT framework operates on multiple aligned modalities, including depth maps, semantic segmentation maps, and pose descriptors.  These modalities are derived from the original JAAD and PIE datasets using standard preprocessing pipelines (depth estimation, semantic segmentation, and pose extraction models). As the original datasets do not provide all modalities directly, users must generate or obtain these representations using appropriate third-party tools. This repository assumes that modality-aligned inputs are already prepared and focuses on the training, fusion, and evaluation stages of the ADAPT framework. The provided `adapt/data/prepare_clips.py` script can be used to construct dataset manifests based on a user-defined directory structure.

## 10. Configs

- `configs/jaad_beh.yaml`
- `configs/jaad_all.yaml`
- `configs/pie.yaml`

The base config includes the paper hyperparameters:

- Swin-V2-T backbone
- Mamba: 2 layers, `d_state=16`, `d_conv=4`, `expand=2`
- SCMA: `top_k=2`
- TFF: 4 layers, 8 heads, MLP ratio `4`, encoder dropout `0.1`, classifier dropout `0.3`
- total batch size `32` via `8/GPU × 4 GPUs`
- Adam, no weight decay, gradient clip `1.0`, early stopping patience `25` 

## 11. Training

Single node, 4 GPUs:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/jaad_beh.yaml
```

PIE:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/pie.yaml
```

### 12. Fine-tuning schedule

- Stage 1, epochs `1–20`: VFE frozen, train non-backbone modules only, `lr=2e-5`
- Stage 2, epochs `21–80`: unfreeze last two Swin stages, backbone lr `2e-6`
- Stage 3, epochs `81–150`: continue fine-tuning, decay all lrs by `0.1` at epoch `100` 

The paper contains a freeze-policy inconsistency: Sec. 2.2 says patch embedding and the first two Swin stages stay frozen throughout, while Sec. 2.6 says all stages are unfrozen in Stage 3. This repository defaults to the stricter Sec. 2.2 interpretation, controlled by `train.schedule.strict_vfe_freeze_rule`.

## 13. Evaluation

```bash
python scripts/eval.py --config configs/jaad_beh.yaml --checkpoint outputs/adapt_jaad_beh/best.pt --split test
```

Metrics implemented exactly as described in the paper:

- Accuracy
- AUC
- F1
- Precision
- Recall
- threshold `0.5` for thresholded metrics

## 14. Inference

```bash
python scripts/infer.py \
  --config configs/pie.yaml \
  --checkpoint outputs/adapt_pie/best.pt \
  --manifest manifests/pie_test.json \
  --output outputs/adapt_pie/test_predictions.json
```

The inference output contains probability, predicted label, and the SCMA sparsity mask for interpretation.

## 15. Reproducibility 

- global seed control for Python, NumPy, and PyTorch
- deterministic PyTorch mode with cuDNN determinism enabled
- config snapshots in checkpoints
- TensorBoard and JSONL metric logs
- best and last checkpoint saving
- DDP-ready with SyncBatchNorm and AMP

## 16. Expected results 

| Split | Acc | AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| JAADBEH | 0.74 | 0.70 | 0.83 | 0.76 | 0.92 |
| JAADALL | 0.91 | 0.85 | 0.76 | 0.74 | 0.78 |
| PIE | 0.92 | 0.90 | 0.83 | 0.84 | 0.81 |

Reported in the paper tables. 

---

## 17. License & Contributions

**License :**
Released under the MIT License. © 2026 ADAPT Authors. All rights reserved.

**Contribution Guidelines :**
We welcome pull requests and improvements.

---
