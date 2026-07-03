from pathlib import Path

import numpy as np
from PIL import Image

from field_segmentation.data.dataset import Torso21Dataset


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

    dataset = Torso21Dataset(root=root, subset="train", split_file=split_file)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["filename"] == filename
    assert sample["image"].size == (5, 4)
    assert sample["mask"].dtype == np.uint8
    assert sample["mask"][0, 0] == 1
    assert sample["mask"][0, 1] == 1
    assert sample["mask"][1, 1] == 0
