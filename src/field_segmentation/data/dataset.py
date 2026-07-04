from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from field_segmentation.data.splits import load_split_file
from field_segmentation.data.transforms import SegmentationTransform


@dataclass(frozen=True)
class Torso21SamplePaths:
    """Paths for one TORSO21 image/mask pair."""

    filename: str
    image_path: Path
    mask_path: Path


class Torso21Dataset(Dataset[dict[str, Any]]):
    """TORSO21 dataset for binary field segmentation.

    The physical dataset layout is expected to be:

    * ``<root>/train/images``
    * ``<root>/train/segmentations``
    * ``<root>/test/images``
    * ``<root>/test/segmentations``

    The expected usage is:

    * pass a local ``split_file`` for derived ``train`` and ``val`` splits
    * omit ``split_file`` for the native dataset ``test`` split

    When ``split_file`` is omitted, the dataset falls back to the native image
    list file if available (``train_images.txt`` or ``test_images.txt``).
    Directory enumeration is only a last-resort fallback for ad hoc inspection.
    """

    def __init__(
        self,
        root: Path,
        subset: str,
        transform: SegmentationTransform,
        split_file: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.split_file = Path(split_file) if split_file is not None else None
        self.transform = transform
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        mask = self.load_binary_mask(sample.mask_path)

        image, mask = self.transform.apply(image, mask)

        return {
            "filename": sample.filename,
            "image": image,
            "mask": mask,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }

    def _build_samples(self) -> list[Torso21SamplePaths]:
        subset_dir = self.root / self.subset
        images_dir = subset_dir / "images"
        masks_dir = subset_dir / "segmentations"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Segmentations directory not found: {masks_dir}")

        filenames = self._load_filenames()
        samples: list[Torso21SamplePaths] = []
        for filename in filenames:
            image_path = images_dir / filename
            mask_path = self._resolve_mask_path(masks_dir=masks_dir, filename=filename)
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image listed in split does not exist: {image_path}"
                )
            samples.append(
                Torso21SamplePaths(
                    filename=filename,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        return samples

    def _load_filenames(self) -> list[str]:
        if self.split_file is not None:
            return load_split_file(self.split_file)

        native_split = self.root / f"{self.subset}_images.txt"
        if native_split.exists():
            return load_split_file(native_split)

        images_dir = self.root / self.subset / "images"
        return sorted(path.name for path in images_dir.iterdir() if path.is_file())

    @staticmethod
    def _resolve_mask_path(masks_dir: Path, filename: str) -> Path:
        direct_match = masks_dir / filename
        if direct_match.exists():
            return direct_match

        stem_matches = sorted(masks_dir.glob(f"{Path(filename).stem}.*"))
        if len(stem_matches) == 1:
            return stem_matches[0]
        if not stem_matches:
            raise FileNotFoundError(
                f"Mask listed in split does not exist for filename: {filename}"
            )
        raise FileNotFoundError(
            f"Multiple mask candidates found for filename {filename}: {stem_matches}"
        )

    @staticmethod
    def load_binary_mask(mask_path: Path) -> np.ndarray:
        """Map any non-black segmentation pixel to the field class.

        The downloaded segmentations currently encode field and field lines with
        non-zero RGB values and background as black. Treating any non-black
        pixel as ``field`` correctly collapses ``field`` and ``line`` into the
        positive class.
        """

        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        return np.any(mask > 0, axis=-1).astype(np.uint8)
