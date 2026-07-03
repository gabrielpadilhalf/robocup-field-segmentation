from pathlib import Path


def test_base_config_exists() -> None:
    assert Path("configs/base.yaml").exists()

