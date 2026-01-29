# YOLOX Training Guide

This guide explains how to train the YOLOX model using the existing codebase.

> **IMPORTANT**: The current training script (`yolox/train.py`) is hardcoded to look for validation dataset paths (`val2017`). To train on your custom dataset without modifying the code, you **must** structure your training data to mimic the COCO validation set structure.

## 1. Dataset Preparation

To satisfy the script's hardcoded paths, organize your dataset exactly as follows:

```text
dataset_root/
├── annotations/
│   └── instances_val2017.json  <-- PLACE YOUR TRAINING ANNOTATIONS HERE
└── val2017/                    <-- PLACE YOUR TRAINING IMAGES HERE
    ├── image01.jpg
    ├── image02.jpg
    └── ...
```

-   **`annotations/instances_val2017.json`**: This file should contain your **training** annotations in COCO format. Do not be confused by the name "val2017"; the script reads this specific filename for training data.
-   **`val2017/`**: This directory should contain all your **training** images.

## 2. Environment Setup

Ensure you have MindSpore installs. You also need `pycocotools`.

```bash
pip install pycocotools
# Install MindSpore based on your hardware (GPU/Ascend)
```

## 3. Training Command

Run the training script from the root of the repository:

```bash
python yolox/train.py \
    --data_dir /path/to/your/dataset_root \
    --name yolox-tiny \
    --device_target GPU \
    --per_batch_size 8 \
    --lr 0.001
```

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data_dir` | Path to the directory containing `val2017` and `annotations`. | `/home/psy...` |
| `--name` | Model name (e.g., `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l`). | `yolox-tiny` |
| `--device_target` | Hardware target: `GPU` or `Ascend`. | `GPU` |
| `--per_batch_size` | Batch size per device. | `4` |
| `--lr` | Learning rate. | `0.0001` |
| `--max_epoch` | Number of training epochs. | `200` |
| `--resume` | Resume training from a checkpoint (`True`/`False`). | `False` |
| `--pretrain_model` | Path to a pretrained checkpoint to load. | `./pretrain_model/...` |

## 4. Output

-   **Logs**: Training logs will be saved in `logs/train/`.
-   **Checkpoints**: Model checkpoints will be saved in `./save/yolox/` (or the path specified by `--save_dir`).

## 5. Troubleshooting

-   **"File not found" errors**: Double-check that your annotation file is named exactly `instances_val2017.json` and is inside an `annotations` folder within your `--data_dir`. Check that images are in `val2017`.

-   **Out of Memory (OOM)**: Reduce `--per_batch_size`.

## 6. CPU Training (No GPU/Ascend)

If you do not have a GPU or Ascend processor, you can train on CPU, but please be aware of the following:

-   **Performance**: Training will be **extremely slow**. It is not recommended for full training runs, but can be used for debugging or testing code logic.
-   **Command**: Change the device target to `CPU`.

```bash
python yolox/train.py \
    --data_dir /path/to/your/dataset_root \
    --name yolox-tiny \
    --device_target CPU \
    --per_batch_size 2 \
    --lr 0.001
```

## 7. Inference / Evaluation

To use your trained model for evaluation (calculating mAP), use `yolox/test.py`.

### Command

```bash
python yolox/test.py \
    --val_data_dir /path/to/your/dataset_root \
    --val_ckpt ./save/yolox/yolox-tiny_1-1_10.ckpt \
    --name yolox-tiny \
    --device_target CPU \
    --conf_thre 0.25 \
    --nms_thre 0.45
```

> **Note**: This uses `val_data_dir` instead of `data_dir` and also expects the same `annotations/instances_val2017.json` structure inside it.

### Key Arguments

| Argument | Description |
| :--- | :--- |
| `--val_ckpt` | Path to your trained checkpoint file. Look in the `save/yolox` folder. |
| `--val_data_dir` | Directory containing `val2017` and `annotations`. |
| `--conf_thre` | Confidence threshold (default 0.001, recommend 0.25+ for visualizing). |
| `--nms_thre` | NMS threshold (default 0.65, recommend 0.45). |

### Note on "Testing" vs "Inference"

The `test.py` script is designed to run validation on a dataset (calculating mAP). If you want to run inference on a single image or a folder of images without ground truth labels, you may need to write a custom script or adapt `test.py` to skip the metric calculation part.
