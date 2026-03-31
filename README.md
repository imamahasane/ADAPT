# ADAPT: Adaptive Cross-Modal Fusion with Sparse Attention for Pedestrian Crossing Intention Prediction

Research-grade PyTorch implementation of **ADAPT**, following the paper *Adaptive Cross-Modal Fusion with Sparse Attention for Pedestrian Crossing Intention Prediction*. The repository implements the five-module pipeline described in the paper: shared **Swin Transformer V2 Tiny** visual encoding, **CMGA** local/global visual fusion, **MFE** with two-layer Mamba SSM, **SCMA** top-k sparse cross-modal attention, and **TFF** ViT-style temporal fusion. fileciteturn2file0L78-L95

## Paper summary

ADAPT predicts whether a pedestrian will cross within a 1–2 second horizon from a 16-frame observation window. Inputs are four aligned visual modalities plus speed, bounding boxes, and pose descriptors. The model uses selective sparse fusion rather than dense inter-modal attention, and the paper reports strong results on JAAD and PIE. fileciteturn2file0L113-L123 fileciteturn2file0L216-L231

## Architecture

```text
RGB/local depth/global semantic/global depth
    │
    ├── shared Swin-V2-T backbone (VFE)
    │      └── 1×1 projection → [B,N,256,8,8]
    │
    ├── CMGA
    │      ├── local branch: RGB + local depth
    │      ├── global branch: semantic + global depth
    │      └── routing gate → F_L, F_G
    │
    ├── MFE
    │      └── speed + bbox + pose → 2-layer Mamba → F_M
    │
    ├── SCMA
    │      └── sparse top-k attention over {F_L, F_G, F_M}
    │
    └── TFF
           └── CLS token + 4-layer transformer encoder + MLP head → p(cross)
```

## Installation

```bash
conda create -n adapt python=3.10 -y
conda activate adapt
pip install -e .
```

Tested dependency targets from the paper:

- `torch >= 2.1.0`
- `timm >= 0.9.12`
- `mamba-ssm >= 1.2.0` fileciteturn2file0L311-L325

## Dataset preparation

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

### Preprocessing rules enforced by the code

- clip length `16`
- overlap `80%` (stride `3`)
- retain clips only if `1s <= TTE <= 2s`
- resize all visual modalities to `256×256`
- standardize RGB and semantic maps with ImageNet mean/std
- replicate single-channel depth to 3 channels and normalize the same way
- normalize speed by `60 km/h` for JAAD or `70 km/h` for PIE
- normalize bounding boxes by image size
- use a **consistent** random horizontal flip across all modalities inside a training clip fileciteturn2file0L280-L310

### Important paper ambiguity

The paper does **not** specify how depth maps, semantic maps, or pose descriptors are generated from raw video. This repository therefore assumes those aligned modalities are already available, and does not invent an undocumented preprocessing model. The provided `adapt/data/prepare_clips.py` is intentionally a manifest-builder stub for your own directory layout.

## Configs

- `configs/jaad_beh.yaml`
- `configs/jaad_all.yaml`
- `configs/pie.yaml`

The base config includes the paper hyperparameters:

- Swin-V2-T backbone
- Mamba: 2 layers, `d_state=16`, `d_conv=4`, `expand=2`
- SCMA: `top_k=2`
- TFF: 4 layers, 8 heads, MLP ratio `4`, encoder dropout `0.1`, classifier dropout `0.3`
- total batch size `32` via `8/GPU × 4 GPUs`
- Adam, no weight decay, gradient clip `1.0`, early stopping patience `25` fileciteturn2file0L252-L263 fileciteturn2file0L313-L326

## Training

Single node, 4 GPUs:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/jaad_beh.yaml
```

PIE:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/pie.yaml
```

### Fine-tuning schedule

- Stage 1, epochs `1–20`: VFE frozen, train non-backbone modules only, `lr=2e-5`
- Stage 2, epochs `21–80`: unfreeze last two Swin stages, backbone lr `2e-6`
- Stage 3, epochs `81–150`: continue fine-tuning, decay all lrs by `0.1` at epoch `100` fileciteturn2file0L252-L258

The paper contains a freeze-policy inconsistency: Sec. 2.2 says patch embedding and the first two Swin stages stay frozen throughout, while Sec. 2.6 says all stages are unfrozen in Stage 3. This repository defaults to the stricter Sec. 2.2 interpretation, controlled by `train.schedule.strict_vfe_freeze_rule`.

## Evaluation

```bash
python scripts/eval.py --config configs/jaad_beh.yaml --checkpoint outputs/adapt_jaad_beh/best.pt --split test
```

Metrics implemented exactly as described in the paper:

- Accuracy
- AUC
- F1
- Precision
- Recall
- threshold `0.5` for thresholded metrics; AUC for model selection fileciteturn2file0L327-L356

## Inference

```bash
python scripts/infer.py \
  --config configs/pie.yaml \
  --checkpoint outputs/adapt_pie/best.pt \
  --manifest manifests/pie_test.json \
  --output outputs/adapt_pie/test_predictions.json
```

The inference output contains probability, predicted label, and the SCMA sparsity mask for interpretation.

## Reproducibility guarantees

- global seed control for Python, NumPy, and PyTorch
- deterministic PyTorch mode with cuDNN determinism enabled
- config snapshots in checkpoints
- TensorBoard and JSONL metric logs
- best and last checkpoint saving
- DDP-ready with SyncBatchNorm and AMP

## Expected results from the paper

| Split | Acc | AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| JAADBEH | 0.74 | 0.70 | 0.83 | 0.76 | 0.92 |
| JAADALL | 0.91 | 0.85 | 0.76 | 0.74 | 0.78 |
| PIE | 0.92 | 0.90 | 0.83 | 0.84 | 0.81 |

Reported in the paper tables. fileciteturn2file0L264-L264 fileciteturn2file0L326-L352

## Citation

Please cite the original ADAPT paper if you use this implementation.
