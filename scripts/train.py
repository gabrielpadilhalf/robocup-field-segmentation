from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_segmentation.data.dataset import Torso21Dataset
from field_segmentation.data.transforms import SegmentationTransform
from field_segmentation.models.small_unet import UNet
from field_segmentation.train.trainer import Trainer
from field_segmentation.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--model",
        choices=("unet", "fast_scnn"),
        default="unet",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration file path.",
    )
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "unet":
        return UNet()
    if model_name == "fast_scnn":
        raise NotImplementedError("fast_scnn is not implemented yet.")
    raise ValueError(f"Unsupported model: {model_name}")


def build_dataloaders(config: dict[str, object]) -> tuple[DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset_root = Path(dataset_config["root"])

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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )
    return train_loader, val_loader


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_config = load_config(REPO_ROOT / "configs" / f"{args.model}.yaml")
    if "training" not in model_config:
        raise KeyError(f"Model config for '{args.model}' must define a 'training' section.")
    config["training"] = model_config["training"]
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
