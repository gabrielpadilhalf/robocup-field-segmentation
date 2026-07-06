from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_segmentation.data.splits import create_val_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic train/validation split for TORSO21.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(".local/dataset/reality"),
        help="Path to the dataset root containing train/ and test/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/splits"),
        help="Directory where train.txt, val.txt, and metadata.json will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of sequence groups reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for grouped split generation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = create_val_split(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"seed={result.seed}")
    print(f"val_ratio={result.val_ratio}")
    print(f"train_groups={len(result.train_groups)}")
    print(f"val_groups={len(result.val_groups)}")
    print(f"train_images={len(result.train_filenames)}")
    print(f"val_images={len(result.val_filenames)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
