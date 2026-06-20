# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch implementation of **PF-Net: Point Fractal Network for 3D Point Cloud Completion** (CVPR 2020). Given a *partial* point cloud, the network predicts only the **missing region** (not the whole shape) using a multi-resolution encoder and a coarse-to-fine pyramid decoder, trained adversarially.

## Environment

- Python 3.7, PyTorch 1.0.1, torchvision. The code uses pre-1.5 idioms (`torch.autograd.Variable`, `.data`, manual `.cpu().detach()`), so newer PyTorch may need small fixes.
- A CUDA GPU is effectively required: `Train_PFNet.py` hardcodes `USE_CUDA = True` and wraps both models in `nn.DataParallel` regardless of the `--cuda` flag. Running on CPU requires editing the script.

## Commands

There is no build / lint / unit-test setup. Scripts are run directly and configured via argparse flags — and, in several spots, by editing constants in the file (e.g. `class_choice`, `infile`).

```bash
# Train (GAN = generator + discriminator). --D_choose 0 trains generator only.
python Train_PFNet.py --batchSize 24 --crop_point_num 512 --D_choose 1

# Chamfer-Distance metrics (CD, Pred->GT, GT->Pred) for one category
python show_CD.py --netG Checkpoint/point_netG.pth        # category set by class_choice in the file

# Generate completion results as .txt point clouds (open in Meshlab)
python show_recon.py --netG Checkpoint/point_netG.pth

# Complete a single partial cloud from CSV (inputs in test_one/)
python Test_csv.py
```

Key training flags: `--crop_point_num` = number of missing points to predict (**must be a multiple of 128** — the decoder reshapes by `crop_point_num/128`); `--point_scales_list` = the 3 input resolutions; `--wtl2` = reconstruction-vs-adversarial weight; resume with `--netG/--netD`.

Checkpoints are saved every 10 epochs to `Trained_Model/` when `--D_choose 1`, else to `Checkpoint/`. **`Trained_Model/` does not exist in the repo — create it before training with the discriminator.** Training loss is appended to `loss_PFNet.txt`.

`self-test/` holds standalone MLP / MR-CMLP ablation variants, each with its own train/eval scripts (their `root=` paths are relative to that subdirectory: `../../dataset/...`).

## Dataset — path gotcha

Uses the **original** ShapeNet part-segmentation benchmark `shapenetcore_partanno_segmentation_benchmark_v0` (per-category `points/*.pts` + `points_label/*.seg` layout — **not** the `_normal` variant with flat `.txt` files).

- `shapenet_part_loader.PartDataset`'s default root is correct: `dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/`.
- **But every entry script (`Train_PFNet.py`, `show_CD.py`, `show_recon.py`, `Test_csv.py`) overrides it with `root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/'`** — missing the `shapenet_part/` segment. As written they will not find the data; fix the `root=` argument (or symlink/rely on the loader default) before running.
- Stanford's original download link (`dataset/download_shapenet_part16_catagories.sh`) is dead. The ~797 MB dataset is gitignored and published as a GitHub release asset (`dataset-v0` → `shapenetcore_partanno_segmentation_benchmark_v0.zip`); see README for the download command.

## Architecture

Pipeline: a partial cloud is downsampled by farthest-point sampling into 3 resolutions → encoder fuses them into one latent vector → decoder emits the missing region at 3 coarse-to-fine levels → loss is multi-scale Chamfer + adversarial.

- **`model_PFNet.py`**
  - `Convlayer` — PointNet-style per-resolution extractor; max-pools intermediate features (128/256/512/1024) and concatenates → 1920-d.
  - `Latentfeature` (CMLP encoder) — one `Convlayer` per input resolution (`point_scales_list`), concatenated across scales and fused by a Conv1d into a single latent vector.
  - `_netG` (generator / Point Pyramid Decoder) — produces `pc1_xyz` (64 "center1"), `pc2_xyz` (128 "center2"), `pc3_xyz` (`crop_point_num`, fine). Coarser predictions are broadcast-added into finer ones (folding-style residual). **Returns all three levels**; training supervises each (weights `alpha1/alpha2` are scheduled by epoch).
  - `_netlocalD` — discriminator over the predicted missing region only.
  - `PointcloudCls` — auxiliary classification head (used with `ModelNet40Loader`).
- **`utils.py`** — Chamfer distance (`PointLoss` for training; `PointLoss_test` also returns the two directional distances), `farthest_point_sample`, `index_points`, `distance_squre`.
- **`shapenet_part_loader.py`** — `PartDataset`; normalizes each cloud to the unit sphere; `classification=True` returns `(points, cls)`. It does **not** precompute the missing region.
- **Cropping** (done inside the train/eval loops, not the loader): pick a random viewpoint from a fixed set of 5 directions; the `crop_point_num` points nearest that viewpoint become the ground-truth missing region and are zeroed out of the input.
