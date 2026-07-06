from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_segmentation.data.dataset import Torso21Dataset
from field_segmentation.data.transforms import SegmentationTransform
from field_segmentation.models.fast_scnn import FastSCNN
from field_segmentation.models.small_unet import SmallUNet
from field_segmentation.train.trainer import Trainer
from field_segmentation.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--model",
        choices=("small_unet", "fast_scnn"),
        default="small_unet",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration file path.",
    )
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "small_unet":
        return SmallUNet()
    if model_name == "fast_scnn":
        return FastSCNN()
    raise ValueError(f"Unsupported model: {model_name}")


def build_dataloaders(config: dict[str, object]) -> tuple[DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset_root = Path(dataset_config["root"])
    num_workers = training_config["num_workers"]
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_dataset = Torso21Dataset(
        root=dataset_root,
        subset=dataset_config["train_subset"],
        transform=SegmentationTransform(config, train=True),
        split_file=Path(dataset_config["train_split_file"]),
    )
    val_dataset = Torso21Dataset(
        root=dataset_root,
        subset=dataset_config["val_subset"],
        transform=SegmentationTransform(config, train=False),
        split_file=Path(dataset_config["val_split_file"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_config = load_config(REPO_ROOT / "configs" / f"{args.model}.yaml")
    if "training" not in model_config:
        raise KeyError(f"Model config for '{args.model}' must define a 'training' section.")
    if "model" not in model_config:
        raise KeyError(f"Model config for '{args.model}' must define a 'model' section.")
    config["training"] = model_config["training"]
    config["model"] = model_config["model"]
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(args.model)
    trainer = Trainer(
        model_config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
