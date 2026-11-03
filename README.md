# PRISM-IR: All-in-One Image Restoration

Code for training, inference, and evaluation on ten image restoration tasks.

## Framework Overview

![Framework overview placeholder](assets/framework.svg)

<!-- Replace assets/framework.svg with the framework figure. -->

## Visual Comparison

![Visual comparison placeholder](assets/comparison.svg)

<!-- Replace assets/comparison.svg with the visual comparison figure. -->

## Setup

```bash
conda create -n irdiff python=3.10
conda activate irdiff
pip install -r requirements-training.txt
```

Keep the provided dependency versions, especially `diffusers` and `peft`.

## Checkpoints

**Our checkpoints — Google Drive: to be added.**

<!-- Replace with: [Google Drive](https://drive.google.com/drive/folders/YOUR_FOLDER_ID) -->

Download [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) and
[DA-CLIP](https://github.com/Algolzw/daclip-uir), then update the `/path/to/...`
entries in `configs/ir.yaml`. Training also requires the CLIP ViT-B/32, VGG16,
and LPIPS VGG calibration files specified there. Weight files are downloaded separately.

## Data Preparation

Set the training and validation roots in `configs/ir.yaml`. Each root uses:

```text
<dataset_root>/
├── noisy/{LQ,GT}/
├── rainy/{LQ,GT}/
├── hazy/{LQ,GT}/
├── low-light/{LQ,GT}/
├── motion-blurry/{LQ,GT}/
├── snowy/{LQ,GT}/
├── raindrop/{LQ,GT}/
├── shadowed/{LQ,GT}/
├── jpeg-compressed/{LQ,GT}/
└── uncompleted/{LQ,GT}/
```

## Training

Initial training uses 256×256 patches, batch size 16, learning rate 1e-4, and no gradient accumulation.
Training uses a cosine learning-rate schedule for 200,000 steps; mixed precision is not forced.
Update the paths in `configs/ir.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_training.sh
```

Training starts from SD-Turbo with freshly initialized adapters. To resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_training.sh \
  --resume training_output/checkpoints/latest_resume.pt
```

## Inference

Set `test_options.pretrained_path` to the downloaded checkpoint.

```bash
# All ten tasks
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh --output_dir outputs

# A single image or folder
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh \
  --input /path/to/images --output_dir outputs_custom
```

Use `--tasks delowlight` to select a task, or `--tile_size 0` for whole-image inference.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash run_evaluate.sh \
  -i outputs -r /path/to/datasets/all_in_one/val \
  --metrics psnr ssim lpips fid clipiqa niqe musiq arniqa \
  --output outputs/metrics.json
```

Results include metrics for each task and their average. With all ten tasks, the task average is the ten-task mean.

## Acknowledgements

This code builds on [S3Diff](https://github.com/ArcticHare105/S3Diff),
[DA-CLIP](https://github.com/Algolzw/daclip-uir),
[OpenCLIP](https://github.com/mlfoundations/open_clip), and
[Vision-aided GAN](https://github.com/nupurkmr9/vision-aided-gan).
