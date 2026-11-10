# EAGG-UNet

Semantic segmentation network for multi-band remote-sensing imagery.

EAGG-UNet is a U-Net backbone with three additions:

1. **Edge enhancement** in every encoder stage (fixed, trainable Sobel-type edge
   kernel + channel and spatial attention).
2. **A graph bottleneck** whose adjacency matrix is generated on the fly from a
   learnable Gaussian kernel over *both* pixel coordinates and learned feature
   similarity, then fused by a learnable weight.
3. **GRRM** (Graph Residual Refinement Module) after the GCN to counter graph
   over-smoothing: channel re-calibration + depthwise local refinement with a
   learnable residual scale.

Verified parameter count: **45,348,432** trainable parameters (20 classes, 5 input bands).

---

## Repository layout

```
EAGG-UNET/
├── configs/
│   ├── train.json          # training configuration
│   └── eval.json           # evaluation configuration
├── models/
│   ├── __init__.py         # exports EAGG + build_model(config)
│   └── eagg.py             # the network (all modules live here)
├── utils/
│   ├── __init__.py
│   └── unet_dataset.py     # GeoTIFF reader, dataset, normalisation stats
├── metrics.py              # confusion matrix, per-class IoU / P / R / F1
├── train.py                # training entry point
├── eval.py                 # evaluation entry point
├── requirements.txt
└── .gitignore
```

This layout is a reorganised version of the original working scripts:

| Original file            | In this repo            | Notes |
|--------------------------|-------------------------|-------|
| `models/EAGG.py`         | `models/eagg.py`        | class `UNetWithFeatureSelect` renamed to `EAGG`; `UNetWithFeatureSelect` kept as an alias |
| `trainy.py`              | `train.py`              | multi-model baseline branches removed, EAGG only |
| `eval.py`                | `eval.py`               | duplicate/dead branches removed, EAGG only |
| `metrics.py`             | `metrics.py`            | unchanged behaviour |
| `utils/unet_dataset.py`  | `utils/unet_dataset.py` | normalisation stats and `to_tensor` centralised |

---

## Architecture

```
input [B, 5, H, W]
  │
  ├─ Encoder1 ( 64ch) ──► skip x1
  ├─ Encoder2 (128ch) ──► skip x2
  ├─ Encoder3 (256ch) ──► skip x3
  ├─ Encoder4 (512ch) ──► skip x4
  │        each Encoder = conv-BN-ReLU ×2 → EdgeEnhance → maxpool
  │
  ├─ Bottleneck: 512 → 1024 ch
  ├─ AdaptiveAvgPool → 32×32 graph grid
  ├─ GenAdj_Gaussian  → A ∈ R^{B×1024×1024}
  ├─ GCN(1024 → 2048 → 1024) with LayerNorm, GELU, residual
  ├─ GRRM (GraphResidualRefinement, 1024 ch)
  ├─ bilinear upsample back to bottleneck resolution
  │
  ├─ Decoder1 (512) ◄─ x4
  ├─ Decoder2 (256) ◄─ x3
  ├─ Decoder3 (128) ◄─ x2
  ├─ Decoder4 ( 64) ◄─ x1
  └─ 1×1 conv → [B, num_classes, H, W]
```

| Module | Class | Purpose |
|---|---|---|
| Encoder / Decoder | `Encoder`, `Decoder` | U-Net stages; encoder returns `(features, pooled)` for skip connections |
| Edge enhancement | `EdgeEnhance` | Sobel-initialised depthwise edge response + channel & spatial attention |
| Adjacency generator | `GenAdj_Gaussian` | Gaussian kernel over spatial distance and feature cosine distance, fused by `sigmoid(fusion_weight)`; `σ_spatial` / `σ_feature` are **learnable** |
| Graph block | `GCN` (`GraphConvolution`) | two graph convolutions `A·(XW)` with LayerNorm, GELU, dropout, residual |
| GRRM | `GraphResidualRefinement` | channel attention + depthwise-separable local refinement + learnable `γ` residual |

Because `σ_spatial` and `σ_feature` are learnable parameters they are stored in
the checkpoint, so a checkpoint carries the adjacency bandwidths it was trained
with. Only shape-defining settings (`num_classes`, `in_channels`,
`gcn_spatial_size`) must match between training and evaluation.

---

## Installation

```bash
git clone <this-repo>
cd EAGG-UNET
pip install -r requirements.txt
```

GDAL is required for GeoTIFF reading. A conda install is usually easiest:

```bash
conda install -c conda-forge gdal
```

## Data preparation

Images are 5-band GeoTIFFs, labels are single-band GeoTIFFs whose values are
class indices (`255` is treated as *ignore*). A data list is a text file with one
sample per line:

```
<image_path>\t<label_path>
```

Paths may be absolute or relative to the list file. Example `newdata/train20.txt`:

```
5bands/A6.tif	20label/A6.tif
5bands/A31.tif	20label/A31.tif
```

Point `train_list`, `test_list` (training) and `img_txt` (evaluation) in the
configs at your own lists.

## Training

```bash
python train.py --config configs/train.json
python train.py --config configs/train.json --save_name eagg_lcma   # custom prefix
```

Outputs, written to `save_model.save_path`:

| File | Content |
|---|---|
| `<name>_best_miou.pth` | weights at the best validation mIoU |
| `<name>_best_loss.pth` | weights at the lowest validation loss |
| `<name>_best_miou.txt`, `<name>_best_loss.txt` | `epoch, loss, mIoU, OA` |
| `<name>_conf_matrix_val.txt` | validation confusion matrix at best mIoU |

A text log is written to `log_dir`. Early stopping triggers when validation mIoU
has not improved for `early_stop_patience` epochs, but never before `min_epochs`.

## Evaluation

```bash
python eval.py --config configs/eval.json
python eval.py --config configs/eval.json --checkpoint saved/models/eagg_lcma_best_miou.pth
```

Prints per-class precision / recall / F1 and the summary:

```
  OA (Overall Accuracy):  ...
  mIoU (Mean IoU):        ...
  Macro F1:               ...
```

The confusion matrix and a results file are written to `result_dir`
(`confuse_matrix/` by default).

## Configuration reference

| Key | Used by | Meaning |
|---|---|---|
| `model_name` | both | checkpoint prefix and display name |
| `num_classes` | both | number of classes (shape-defining) |
| `in_channels` | both | input bands, 5 for the LCMA data (shape-defining) |
| `gcn_spatial_size` | both | graph grid side, 32 → 1024 nodes (shape-defining) |
| `sigma_spatial`, `sigma_feature` | both | initial bandwidths (learnable, stored in the checkpoint) |
| `use_spatial`, `use_feature` | both | which adjacency sources to combine |
| `batch_size`, `lr`, `momentum`, `weight_decay`, `num_epoch` | train | optimisation |
| `train_list`, `test_list` | train | data lists |
| `save_model.save_path` | both | checkpoint directory |
| `early_stop_patience`, `min_epochs` | train | early stopping |
| `num_workers`, `seed`, `device` | train | loader workers, reproducibility, `"auto"`/`"cuda:0"`/`"cpu"` |
| `img_txt`, `checkpoint`, `result_dir` | eval | test list, explicit weights, output directory |

`device: "auto"` selects `cuda:0` when available and falls back to CPU otherwise.

## Troubleshooting

**`ImportError: ... libtiff.so.6: undefined symbol: jpeg12_write_raw_data`** or
**`ModuleNotFoundError: No module named '_gdal'`**

Some conda environments load a `libjpeg` through `torchvision` that is
incompatible with the `libtiff` used by GDAL. Importing `torchvision` *before*
`osgeo` then breaks GDAL. This repo imports the dataset helpers first, so the
entry points are unaffected. If you import modules in your own script, do:

```python
from utils.unet_dataset import read_tiff   # GDAL first
from torchvision import transforms         # then torchvision
```

**Out of memory during training.** The graph adjacency is
`gcn_spatial_size² × gcn_spatial_size²` per sample, so lowering
`gcn_spatial_size` (e.g. 32 → 16) reduces memory quadratically. Lower
`batch_size` as usual.

---

## 中文快速开始

```bash
# 1. 环境
conda install -c conda-forge gdal
pip install -r requirements.txt

# 2. 准备数据列表（每行：图像路径 <tab> 标签路径）
#    图像 5 波段 GeoTIFF，标签单波段，255 为忽略值

# 3. 训练
python train.py --config configs/train.json

# 4. 评估
python eval.py --config configs/eval.json \
    --checkpoint saved/models/EAGG_best_miou.pth
```

注意事项：

- `num_classes`、`in_channels`、`gcn_spatial_size` 属于**影响网络形状**的参数，
  训练和评估的配置必须一致，否则加载权重会报形状不匹配。
- `sigma_spatial` / `sigma_feature` 是**可学习参数**，会存进 checkpoint，
  配置里的值只决定初始化。
- 若在本机自定义脚本里同时使用 torchvision 和 GDAL，请**先导入 GDAL**
  （见上方 Troubleshooting）。

## License / Citation

No license file is bundled yet — add one (e.g. MIT or Apache-2.0) before making
the repository public. Add the paper citation here once available.
