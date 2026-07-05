from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_segmentation.data.dataset import Torso21Dataset
from field_segmentation.data.transforms import SegmentationTransform
from field_segmentation.eval.metrics import Metrics, eval_model
from field_segmentation.models.fast_scnn import FastSCNN
from field_segmentation.models.small_unet import SmallUNet
from field_segmentation.utils.config import load_config


MODEL_NAMES = ("small_unet", "fast_scnn")


def build_test_loader(config: dict[str, object]) -> tuple[DataLoader, Torso21Dataset]:
    dataset_config = config["dataset"]
    test_config = config["test"]
    dataset = Torso21Dataset(
        root=Path(dataset_config["root"]),
        subset=dataset_config["test_subset"],
        transform=SegmentationTransform(config, train=False),
        split_file=Path(dataset_config["test_split_file"]),
    )
    num_workers = test_config["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=test_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def benchmark_cpu_forward(
    model: torch.nn.Module,
    dataset: Torso21Dataset,
    discard_first: int,
    num_images: int,
) -> dict[str, float | int | str]:
    model.eval()
    total_seconds = 0.0
    total_images = 0
    warmup_end_index = min(len(dataset), discard_first)
    timed_end_index = min(len(dataset), warmup_end_index + num_images)
    with torch.no_grad():
        for index in range(warmup_end_index):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).cpu()
            model(image)

        for index in range(warmup_end_index, timed_end_index):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).cpu()
            start_time = time.perf_counter()
            model(image)
            total_seconds += time.perf_counter() - start_time
            total_images += 1

    if total_images == 0:
        raise ValueError("No images available for CPU timing benchmark.")

    return {
        "device": "cpu",
        "discarded_image_count": discard_first,
        "image_count": total_images,
        "average_image_seconds": total_seconds / total_images,
        "total_timed_seconds": total_seconds,
    }


def choose_visualization_indices(
    dataset_size: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    if dataset_size == 0:
        return []
    num_samples = min(dataset_size, num_samples)
    generator = random.Random(seed)
    return sorted(generator.sample(range(dataset_size), k=num_samples))


def render_prediction_mask(prediction: torch.Tensor) -> Image.Image:
    mask = prediction.to(torch.uint8).cpu().numpy() * 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def save_prediction_figure(
    model: torch.nn.Module,
    dataset: Torso21Dataset,
    sample_indices: list[int],
    output_path: Path,
) -> None:
    if not sample_indices:
        raise ValueError("No sample indices provided for visualization.")

    rows: list[Image.Image] = []
    model.eval()
    with torch.no_grad():
        for index in sample_indices:
            sample = dataset[index]
            original_image = Image.open(sample["image_path"]).convert("RGB")
            input_tensor = sample["image"].unsqueeze(0).cpu()
            prediction_logits = model(input_tensor)
            prediction_mask = torch.argmax(prediction_logits, dim=1)[0]
            predicted_image = render_prediction_mask(prediction_mask)
            target_size = predicted_image.size
            original_panel = original_image.resize(
                target_size,
                Image.Resampling.BILINEAR,
            )
            row = Image.new("RGB", (target_size[0] * 2, target_size[1]))
            row.paste(original_panel, (0, 0))
            row.paste(predicted_image, (target_size[0], 0))
            rows.append(row)

    figure_width = rows[0].width
    figure_height = sum(row.height for row in rows)
    figure = Image.new("RGB", (figure_width, figure_height))
    current_top = 0
    for row in rows:
        figure.paste(row, (0, current_top))
        current_top += row.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)


def main() -> int:
    config = load_config("configs/base.yaml")
    test_loader, test_dataset = build_test_loader(config)
    evaluation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in MODEL_NAMES:
        if model_name == "small_unet":
            model = SmallUNet()
        elif model_name == "fast_scnn":
            model = FastSCNN()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        checkpoint_path = Path(config["paths"]["checkpoints_dir"]) / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(evaluation_device)

        criterion = torch.nn.CrossEntropyLoss()
        metrics: Metrics = eval_model(
            model,
            criterion,
            test_loader,
            evaluation_device,
            desc=f"Test {model_name}",
        )

        model = model.to(torch.device("cpu"))
        timing = benchmark_cpu_forward(
            model,
            test_dataset,
            discard_first=config["test"]["timing_discard_first"],
            num_images=config["test"]["timing_num_images"],
        )

        results_dir = Path(config["test"]["results_dir"]) / model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / config["test"]["metrics_filename"]).write_text(
            json.dumps(
                {
                    "model": model_name,
                    "iou": metrics.iou,
                    "dice": metrics.dice,
                    "test_image_count": len(test_dataset),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (results_dir / config["test"]["timing_filename"]).write_text(
            json.dumps(
                {
                    "model": model_name,
                    "device": timing["device"],
                    "discarded_image_count": timing["discarded_image_count"],
                    "image_count": timing["image_count"],
                    "average_image_seconds": timing["average_image_seconds"],
                    "total_timed_seconds": timing["total_timed_seconds"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        sample_indices = choose_visualization_indices(
            dataset_size=len(test_dataset),
            num_samples=config["test"]["visualization_samples"],
            seed=config["test"]["visualization_seed"],
        )
        save_prediction_figure(
            model=model,
            dataset=test_dataset,
            sample_indices=sample_indices,
            output_path=results_dir / config["test"]["figure_filename"],
        )

        print(
            f"{model_name}: iou={metrics.iou:.4f} dice={metrics.dice:.4f} "
            f"avg_cpu_image_s={timing['average_image_seconds']:.6f}"
        )
        print(f"{model_name}: results saved to {results_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
