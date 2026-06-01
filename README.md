# TSViT Crop-Type Semantic Segmentation

This repository contains the group's reproduction and analysis of **TSViT for crop-type semantic segmentation from satellite image time series**. The project focuses on the PASTIS24 benchmark and evaluates the model with both quantitative metrics and qualitative segmentation maps.

## Links

- Code repository: `https://github.com/loanhviet/tsvit-crop-segmentation`
- Demo GitHub.io: `https://loanhviet.github.io/tsvit-crop-segmentation/`
- Demo source: [docs/index.html](docs/index.html)
- Clean Colab notebook: [TSViT.ipynb](TSViT.ipynb)

## Main Questions

| Question | Experiment | Outcome |
|---|---|---|
| CQ1 | TSViT vs UNet3D | TSViT improves mIoU and macro F1 while using fewer parameters. |
| CQ2 | TSViT vs TViT vs STViT | Removing spatial modeling hurts most; temporal-then-spatial performs best. |

The final demo reports benchmark test-set results only. Practical behavior is shown through qualitative prediction maps on the PASTIS24 test split.

## Dataset

The main dataset is **PASTIS24**, a 24x24 patch version of PASTIS for semantic segmentation.

Expected sample structure:

```python
sample = {
    "img":    (T, 10, 24, 24),  # Sentinel-2 time series
    "labels": (3, 24, 24),      # segmentation masks; training uses labels[0]
    "doy":    (T,),             # day-of-year values
}
```

Class `19` is treated as void/unknown and masked during training and evaluation.

## Results

### CQ1: TSViT vs UNet3D

| Model | Loss | OA | mIoU | F1-macro | Precision | Recall | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSViT | 0.6022 | 0.8271 | 0.6361 | 0.7623 | 0.7841 | 0.7465 | 1.657M |
| UNet3D | 0.6281 | 0.8011 | 0.5720 | 0.7045 | 0.7573 | 0.6711 | 6.177M |

### CQ2: Ablation

| Model | Loss | OA | mIoU | F1-macro | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| TSViT full | 0.6022 | 0.8271 | 0.6361 | 0.7623 | 0.7841 | 0.7465 |
| TViT, no spatial | 1.0591 | 0.6685 | 0.3713 | 0.5267 | 0.5654 | 0.5049 |
| STViT, spatial-first | 0.7707 | 0.7939 | 0.5572 | 0.6921 | 0.7424 | 0.6660 |

Delta mIoU vs TSViT full:

| Variant | Delta mIoU | Interpretation |
|---|---:|---|
| TViT | -0.2648 | Spatial modeling is critical for pixel-level segmentation. |
| STViT | -0.0789 | Spatial-first is better than temporal-only, but weaker than TSViT. |

## Reproduction

Set dataset paths in `data/datasets.yaml`, then run the training scripts.

```bash
# TSViT, original repo config
python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_fold1.yaml \
  --device 0

# Colab/A100 configs used for the submitted experiments
python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_fold1_colab_full_a100_opt.yaml --device 0
python train_and_eval/segmentation_training.py --config configs/PASTIS24/UNet3D_fold1_colab_full_a100_opt.yaml --device 0
python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TViT_fold1_colab_full_a100_opt.yaml --device 0
python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/STViT_fold1_colab_full_a100_opt.yaml --device 0
```

For Colab runs, use [TSViT.ipynb](TSViT.ipynb). The notebook is cleaned for submission: execution outputs are removed, long-running train/eval cells are guarded by flags, and dataset/checkpoint artifacts are kept outside GitHub.

## Demo

The GitHub Pages demo is a static report:

```text
docs/
  index.html
  assets/
```

Publish with GitHub Pages:

1. Push this repository to `https://github.com/loanhviet/tsvit-crop-segmentation`.
2. Open `Settings -> Pages`.
3. Select `Deploy from a branch`.
4. Choose branch `main` and folder `/docs`.

## Citation

This project is based on the original DeepSatModels implementation:

- Tarasiou, M., Chavez, E., Zafeiriou, S. "ViTs for SITS: Vision Transformers for Satellite Image Time Series", CVPR 2023.
- Original repository: https://github.com/michaeltrs/DeepSatModels

The original project is licensed under Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
