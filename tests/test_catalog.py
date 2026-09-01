from pathlib import Path

import cv2
import numpy as np
import pytest

from petrovision.catalog import build_catalog, modality_pair_table


def _write_image(path: Path, value: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_catalog_and_pairing(tmp_path: Path) -> None:
    _write_image(tmp_path / "PPL" / "train" / "oolite" / "sample_01.jpg")
    _write_image(tmp_path / "XPL" / "train" / "oolite" / "sample_01.jpg")
    _write_image(tmp_path / "PPL" / "train" / "oolite" / "sample_02.jpg")

    catalog = build_catalog(
        tmp_path,
        modes=["PPL", "XPL"],
        splits=["train", "val", "test"],
        extensions=[".jpg"],
    )
    pairs = modality_pair_table(catalog)

    assert len(catalog) == 3
    assert int(pairs["has_pair"].sum()) == 1
    assert set(catalog["class_name"]) == {"oolite"}


def test_catalog_rejects_identity_reused_across_splits(tmp_path: Path) -> None:
    _write_image(tmp_path / "PPL" / "train" / "oolite" / "Oolitic.20.jpg")
    _write_image(tmp_path / "PPL" / "val" / "oolite" / "Oolitic.20.jpg", 130)

    with pytest.raises(ValueError, match="reutilizados entre treino"):
        build_catalog(
            tmp_path,
            modes=["PPL"],
            splits=["train", "val", "test"],
            extensions=[".jpg"],
        )
