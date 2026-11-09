# PRISM-IR

Training and testing for 3-task, 5-task, and 10-task image restoration.

## Installation

```bash
conda create -n prism-ir python=3.10
conda activate prism-ir
pip install -r requirements-training.txt
```

## Weights and Configuration

Download the weights for the selected setting. Google Drive links will be added below.
[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) is downloaded automatically
when `sd_path` is `null`; a local path can also be used.

| Setting | Config | DA-CLIP (Google Drive) | Model (Google Drive) |
|---|---|---|---|
| 3-task | [ir_3task.yaml](configs/ir_3task.yaml) | To be added | To be added |
| 5-task | [ir_5task.yaml](configs/ir_5task.yaml) | To be added | To be added |
| 10-task | [ir_10task.yaml](configs/ir_10task.yaml) | To be added | To be added |

Use the corresponding filenames from the YAML:

| Setting | DA-CLIP | Model |
|---|---|---|
| 3-task | `daclip_3_tasks.pt` | `checkpoints/model_3_tasks.pkl` |
| 5-task | `daclip_5_tasks.pt` | `checkpoints/model_5_tasks.pkl` |
| 10-task | `daclip_10_tasks.pt` | `checkpoints/model_10_tasks.pkl` |

Set the dataset, DA-CLIP, and model paths in the selected YAML.
Paired datasets use `<dataset_root>/<task_folder>/{LQ,GT}`.
All training and testing settings are read from the YAML.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 bash run_training.sh --base_config configs/ir_3task.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 bash run_training.sh --base_config configs/ir_5task.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 bash run_training.sh --base_config configs/ir_10task.yaml
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh --base_config configs/ir_3task.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh --base_config configs/ir_5task.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh --base_config configs/ir_10task.yaml
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash run_evaluate.sh \
  -i /path/to/outputs -r /path/to/GT
```

## Acknowledgements

[S3Diff](https://github.com/ArcticHare105/S3Diff),
[DA-CLIP](https://github.com/Algolzw/daclip-uir),
[OpenCLIP](https://github.com/mlfoundations/open_clip), and
[Vision-aided GAN](https://github.com/nupurkmr9/vision-aided-gan).
[S3Diff license](src/S3Diff_LICENSE),
[OpenCLIP license](src/open_clip/LICENSE), and
[Vision-aided GAN license](src/vision_aided_loss/LICENSE).
