from pathlib import Path

from field_segmentation.data.splits import build_grouped_split, create_val_split, extract_group_id


def test_extract_group_id_supports_known_patterns() -> None:
    assert extract_group_id("1028-img00145.png") == "1028"
    assert extract_group_id("1037-image00022.png") == "1037"
    assert extract_group_id("1067-frame10.jpg") == "1067"
    assert (
        extract_group_id("97-26_01_2018__18_40_34_0785_upper.png")
        == "97-26_01_2018__18_40_34"
    )


def test_build_grouped_split_keeps_groups_disjoint() -> None:
    filenames = [
        "1028-img00145.png",
        "1028-img00162.png",
        "1029-img00145.png",
        "1030-image00022.png",
        "1030-image00033.png",
    ]
    result = build_grouped_split(filenames=filenames, val_ratio=0.34, seed=7)
    train_groups = {extract_group_id(name) for name in result.train_filenames}
    val_groups = {extract_group_id(name) for name in result.val_filenames}
    assert train_groups.isdisjoint(val_groups)
    assert sorted(result.train_filenames + result.val_filenames) == sorted(filenames)


def test_build_grouped_split_targets_image_ratio_not_group_ratio() -> None:
    filenames = [
        "1000-img0001.png",
        "1000-img0002.png",
        "1000-img0003.png",
        "1000-img0004.png",
        "1001-img0001.png",
        "1002-img0001.png",
        "1003-img0001.png",
    ]
    result = build_grouped_split(filenames=filenames, val_ratio=0.2, seed=0)

    assert len(result.val_groups) == 1
    assert len(result.val_filenames) == 1
    assert len(result.val_filenames) / len(filenames) < 0.3


def test_create_val_split_is_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "segmentations"
    train_images.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    for filename in (
        "1028-img00145.png",
        "1028-img00162.png",
        "1029-img00145.png",
        "1030-image00022.png",
    ):
        (train_images / filename).write_bytes(b"image")
        (train_masks / filename).write_bytes(b"mask")

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = create_val_split(dataset_root=dataset_root, output_dir=first_output, seed=42)
    second = create_val_split(dataset_root=dataset_root, output_dir=second_output, seed=42)

    assert first.train_filenames == second.train_filenames
    assert first.val_filenames == second.val_filenames
    assert (first_output / "train.txt").read_text(encoding="utf-8") == (
        second_output / "train.txt"
    ).read_text(encoding="utf-8")


def test_create_val_split_writes_counts_close_to_requested_ratio(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "segmentations"
    train_images.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    for filename in (
        "1000-img0001.png",
        "1000-img0002.png",
        "1000-img0003.png",
        "1000-img0004.png",
        "1001-img0001.png",
        "1002-img0001.png",
        "1003-img0001.png",
    ):
        (train_images / filename).write_bytes(b"image")
        (train_masks / filename).write_bytes(b"mask")

    result = create_val_split(dataset_root=dataset_root, output_dir=tmp_path / "out", seed=0)

    assert len(result.val_filenames) == 1
