from pathlib import Path

import numpy as np
import torch
from PIL import Image

from field_segmentation.data.dataset import Torso21Dataset
from field_segmentation.data.transforms import SegmentationTransform


def test_dataset_reads_split_and_binarizes_mask(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images_dir = root / "train" / "images"
    masks_dir = root / "train" / "segmentations"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    filename = "1028-img00145.png"
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(images_dir / filename)
    mask = np.zeros((4, 5, 3), dtype=np.uint8)
    mask[0, 0] = [255, 255, 255]
    mask[0, 1] = [254, 254, 254]
    Image.fromarray(mask).save(masks_dir / filename)

    split_file = tmp_path / "val.txt"
    split_file.write_text(f"{filename}\n", encoding="utf-8")

    config = {
        "dataset": {"input_size": [5, 4]},
        "transforms": {
            "prob_horizontal_flip": 0.0,
            "prob_brightness_contrast": 0.0,
            "prob_random_shift_scale_rotate": 0.0,
            "shift_limit": 0.0,
            "scale_limit": 0.0,
            "rotate_limit": 0,
        },
    }
    transform = SegmentationTransform(config, train=False)
    dataset = Torso21Dataset(
        root=root,
        subset="train",
        transform=transform,
        split_file=split_file,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["filename"] == filename
    assert tuple(sample["image"].shape) == (3, 4, 5)
    assert sample["mask"].dtype == torch.long
    assert sample["mask"][0, 0] == 1
    assert sample["mask"][0, 1] == 1
    assert sample["mask"][1, 1] == 0
