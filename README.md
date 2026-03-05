# VIMNET (Unofficial) — Spatiotemporal Transformer for Vehicle Interaction Modeling

This repository is a **research-grade, from-scratch implementation** of the paper draft you provided:

> **VIMNET: Spatiotemporal Transformer for Vehicle Interaction Modeling via Trajectory Prediction in Dynamic Traffic**

The goal is to make it easy to reproduce:
- **Datasets:** **pNEUMA** (urban) and **highD** (highway)
- **Experimental settings:** 10 Hz, 2s observation, 5s prediction, ego-frame normalization, Savitzky–Golay denoising, kinematic filtering
- **Ablations:** neighborhood construction (sector / kNN / radius), edge typing & active/passive asymmetry, temporal modeling (spatial-only / temporal-only / GSAN-style), decoder head design, single-task vs multi-task fine-tuning

> ⚠️ This is an *unofficial* implementation and is not affiliated with the dataset authors.

---

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with **PyTorch 2.x**, CUDA 12.

---

## 2) Datasets

### 2.1 highD (recommended first)

Download the **highD** dataset from the official provider and place files under a directory such as:

```
/data/highD/
  01_tracks.csv
  01_tracksMeta.csv
  01_recordingMeta.csv
  ...
```

Update `configs/preprocess/highd.yaml` → `raw_dir`.

### 2.2 pNEUMA

Download pNEUMA “tiles” (CSV trajectory logs) and place them under:

```
/data/pNEUMA/
  tile_01.csv
  tile_02.csv
  ...
```

Update `configs/preprocess/pneuma.yaml` → `raw_dir`, and list your tiles in `configs/splits/pneuma.yaml`.

> **Coordinate note:** Public pNEUMA CSVs are often provided in **latitude/longitude** (degrees).
> Set `coords: auto` (default) in `configs/preprocess/pneuma.yaml` to auto-detect and project to **local meters** for consistent FoV/kinematics.

---

## 3) Preprocessing

### 3.1 highD → shards

```bash
python scripts/preprocess_highd.py --cfg configs/preprocess/highd.yaml
```

Output structure:

```
data/processed/highd/
  train/
    shard_0000.npz
    ...
  val/
  test/
```

### 3.2 pNEUMA → shards

```bash
python scripts/preprocess_pneuma.py --cfg configs/preprocess/pneuma.yaml
```

### 3.3 What preprocessing does

- Resample to **10 Hz**
- Extract sequences with:
  - **observation** = 2.0 s (20 frames)
  - **prediction** = 5.0 s (50 frames)
  - **stride** = 0.5 s
- Denoise with **Savitzky–Golay** (`window=9`, `order=2`)
- Convert to ego-centric **SE(2)** frame (target at origin, heading aligned to +x)
- Filter implausible segments:
  - speed > 55 m/s
  - lateral acceleration > 8 m/s²
- Build a **fixed-slot neighborhood**:
  - `sector`: EV + (LP,P,RP,LF,F,RF,L,R) = **9 slots**
  - `knn`: EV + k nearest neighbors (pad NULL if fewer)
  - `radius`: EV + up to k neighbors inside radius R

Each shard stores:

- `obs`: (B, T_obs, N, 2) ego-frame positions
- `obs_mask`: (B, T_obs, N) valid-token mask (NULL slots are False)
- `fut`: (B, T_pred, 2) future ego positions
- `lc_label`: (B,) ∈ {0,1,2} for Left/Keep/Right

---

## 4) Training

### 4.1 Pretraining (next-step prediction)

Pretraining follows the draft: autoregressive **next-step** offset from a context window.

> You should first create a mixed pretraining corpus (pNEUMA + highD) under:
> `data/processed/mixed_pretrain/{train,val}/`
> (or adjust `configs/train/pretrain.yaml`).

Run:

```bash
python scripts/pretrain.py --cfg configs/train/pretrain.yaml
```

Checkpoint:
- `runs/pretrain/best_pretrain.pt`

### 4.2 Fine-tuning (trajectory + lane-change)

```bash
python scripts/finetune.py --cfg configs/train/finetune_highd.yaml
```

---

## 5) Reproducing Ablations

Ablation configs live under `configs/ablations/`.

### 5.1 Neighborhood construction (pNEUMA table)

Re-run preprocessing for each neighborhood mode:

- `sector` (ours): `configs/preprocess/pneuma.yaml`
- `knn`: set `neighborhood.mode: knn`
- `radius`: set `neighborhood.mode: radius`

Then train/fine-tune with the same training config (pointing to the corresponding processed folder).

### 5.2 Typed edges + active/passive extents (highD table)

```bash
python scripts/finetune.py --cfg configs/ablations/finetune_highd_fixedweights.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_noactpas.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_notypes.yaml
```

### 5.3 Spatiotemporal attention vs spatial/temporal-only vs GSAN-style

```bash
python scripts/finetune.py --cfg configs/ablations/finetune_highd_spatial_only.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_temporal_only.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_gsan_style.yaml
```

### 5.4 Decoder head ablation

```bash
python scripts/finetune.py --cfg configs/ablations/finetune_highd_head_mlp.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_head_transformer.yaml
```

### 5.5 Multi-task vs single-task fine-tuning

```bash
python scripts/finetune.py --cfg configs/ablations/finetune_highd_single_tp.yaml
python scripts/finetune.py --cfg configs/ablations/finetune_highd_single_lc.yaml
# Multi-task baseline is configs/train/finetune_highd.yaml
```

---

## 6) Notes on Differences vs the Draft

The draft describes several biases and architectural knobs. This implementation supports them via config:

- **Typed sector edges:** `use_type_bias`, `use_slot_type_embeddings`
- **Active/passive asymmetry:** `use_actpas_bias`
- **Distance bias in attention:** `use_distance_bias`
- **VIMNET-FixedWeights:** `fixed_spatial_weights`

The attention layer uses a **block-sparse mask**:
- Causal temporal edges within each slot
- EV↔neighbor edges inside each frame

---

## 7) Citation

If you use this code in research, please cite the original datasets (highD, pNEUMA) and the paper draft.
