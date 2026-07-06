from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import re

_GROUP_PATTERNS = (
    re.compile(r"^(?P<group>.+?)-(?:img|image|frame)\d+\.[^.]+$"),
    re.compile(r"^(?P<group>.+?)frame\d+(?:_[^_]+)?\.[^.]+$"),
    re.compile(r"^(?P<group>.+)frame\d+\.[^.]+$"),
    re.compile(r"^(?P<group>.+?)-aufnahme\d+_[^_]+\.[^.]+$"),
    re.compile(r"^(?P<group>.+?) \d{2}:\d{2}:\d{2}(?:\.\d+)?\.[^.]+$"),
    re.compile(r"^(?P<group>.+)-\d+\.[^.]+$"),
    re.compile(r"^(?P<group>.+)_\d+_[^_]+\.[^.]+$"),
    re.compile(r"^(?P<group>.+)_\d+\.[^.]+$"),
)


@dataclass(frozen=True)
class SplitResult:
    """Deterministic train/validation split output."""

    seed: int
    val_ratio: float
    train_filenames: list[str]
    val_filenames: list[str]
    train_groups: list[str]
    val_groups: list[str]


def extract_group_id(filename: str) -> str:
    """Extract a stable sequence/group id from a dataset filename."""

    for pattern in _GROUP_PATTERNS:
        match = pattern.match(filename)
        if match is not None:
            return match.group("group")
    raise ValueError(f"Unsupported filename pattern for split grouping: {filename}")


def load_split_file(path: Path) -> list[str]:
    """Read split filenames from a text file."""

    lines = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    return [line for line in lines if line]


def build_grouped_split(
    filenames: list[str],
    val_ratio: float,
    seed: int,
) -> SplitResult:
    """Split filenames into train/val partitions by sequence group.

    Groups are shuffled deterministically, then added to validation until the
    accumulated image count is as close as possible to the requested ratio.
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if not filenames:
        raise ValueError("Cannot generate split from an empty filename list")

    grouped: dict[str, list[str]] = {}
    for filename in sorted(filenames):
        group_id = extract_group_id(filename)
        grouped.setdefault(group_id, []).append(filename)

    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("At least two groups are required to create train/val splits")

    shuffled_groups = groups[:]
    random.Random(seed).shuffle(shuffled_groups)
    target_val_images = len(filenames) * val_ratio
    val_groups_unsorted: list[str] = []
    accumulated_val_images = 0

    for group_id in shuffled_groups:
        group_size = len(grouped[group_id])
        if not val_groups_unsorted:
            val_groups_unsorted.append(group_id)
            accumulated_val_images += group_size
            continue

        current_distance = abs(accumulated_val_images - target_val_images)
        next_distance = abs((accumulated_val_images + group_size) - target_val_images)
        if next_distance <= current_distance:
            val_groups_unsorted.append(group_id)
            accumulated_val_images += group_size

    train_groups = sorted(
        group_id for group_id in shuffled_groups if group_id not in val_groups_unsorted
    )
    val_groups = sorted(val_groups_unsorted)
    if not train_groups:
        raise ValueError(
            "Validation selection consumed all groups; train split would be empty"
        )

    train_filenames = sorted(
        filename for group_id in train_groups for filename in grouped[group_id]
    )
    val_filenames = sorted(
        filename for group_id in val_groups for filename in grouped[group_id]
    )
    return SplitResult(
        seed=seed,
        val_ratio=val_ratio,
        train_filenames=train_filenames,
        val_filenames=val_filenames,
        train_groups=train_groups,
        val_groups=val_groups,
    )


def create_val_split(
    dataset_root: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """Create and persist a grouped validation split for the physical train subset."""

    train_images_dir = Path(dataset_root) / "train" / "images"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")

    filenames = sorted(
        path.name for path in train_images_dir.iterdir() if path.is_file()
    )
    result = build_grouped_split(filenames=filenames, val_ratio=val_ratio, seed=seed)
    write_split_outputs(output_dir=output_dir, result=result)
    return result


def write_split_outputs(output_dir: Path, result: SplitResult) -> None:
    """Persist split files and metadata under a local output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(output_dir / "train.txt", result.train_filenames)
    _write_lines(output_dir / "val.txt", result.val_filenames)
    metadata = {
        "seed": result.seed,
        "val_ratio": result.val_ratio,
        "train_image_count": len(result.train_filenames),
        "val_image_count": len(result.val_filenames),
        "train_group_count": len(result.train_groups),
        "val_group_count": len(result.val_groups),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_lines(path: Path, filenames: list[str]) -> None:
    path.write_text("\n".join(filenames) + "\n", encoding="utf-8")
