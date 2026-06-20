# PF-Net-Point-Fractal-Network

PyTorch implementation of **PF-Net: Point Fractal Network for 3D Point Cloud Completion** (CVPR 2020).

Paper: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf>

> Fork of [zztianzz/PF-Net-Point-Fractal-Network](https://github.com/zztianzz/PF-Net-Point-Fractal-Network) with the dataset re-hosted as a GitHub release (the original Stanford download is offline) and documentation fixes.

## 0) Environment
- Python 3.7.4
- PyTorch 1.0.1

## 1) Dataset

PF-Net is trained on the original ShapeNet part-segmentation benchmark
(`shapenetcore_partanno_segmentation_benchmark_v0`, the `.pts` / `.seg` layout).
The original Stanford link is no longer reachable, so the dataset is mirrored as a
release asset of this repository:

```bash
cd dataset
wget https://github.com/Wenri/PF-Net-Point-Fractal-Network/releases/download/dataset-v0/shapenetcore_partanno_segmentation_benchmark_v0.zip
mkdir -p shapenet_part
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d shapenet_part/
```

This produces `dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/`,
the default path expected by `shapenet_part_loader.py`.

> **Note:** the train/eval scripts currently pass an explicit
> `root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/'`. Point that
> argument at the extracted folder above (or rely on the `PartDataset` default) before running.

(The legacy `dataset/download_shapenet_part16_catagories.sh` is kept for reference; an
older Baidu mirror was: 链接 https://pan.baidu.com/s/1MavAO_GHa0a6BZh4Oaogug 提取码 3hoe.)

## 2) Train
```bash
python Train_PFNet.py
```
- `--crop_point_num` controls the number of missing points (must be a multiple of 128).
- `--point_scales_list` controls the input resolutions.
- `--D_choose 1` trains with the discriminator (D-Net); `--D_choose 0` trains the generator only.

## 3) Evaluate on ShapeNet
```bash
python show_recon.py   # writes completion results as .txt (view in Meshlab)
python show_CD.py      # reports the Chamfer Distance and the two metrics from the paper
```

## 4) Complete a single CSV point cloud
Incomplete examples live in `test_one/`:
```bash
python Test_csv.py
```
Change `infile` / `infile_real` in the script to select a different example.

## 5) Visualization
Open the generated `.txt` files with [Meshlab](https://www.meshlab.net/).
