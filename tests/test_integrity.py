from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from petrovision.integrity import (
    apply_near_duplicate_reviews,
    find_cross_split_near_duplicates,
    perceptual_hash,
    phash_hamming_distance,
)


def _write_pattern(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64), dtype=np.uint8)
    image[8:28, 10:30] = 220
    image[36:55, 35:58] = 130
    assert cv2.imwrite(str(path), image)


def test_perceptual_hash_matches_same_pixels_in_different_formats(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.tif"
    _write_pattern(first)
    _write_pattern(second)

    distance = phash_hamming_distance(perceptual_hash(first), perceptual_hash(second))

    assert distance == 0


def test_near_duplicate_audit_compares_different_splits(tmp_path: Path) -> None:
    train_path = Path("data/raw/PPL/train/class17_oolite/train.png")
    val_path = Path("data/raw/PPL/val/class17_oolite/val.tif")
    _write_pattern(tmp_path / train_path)
    _write_pattern(tmp_path / val_path)
    manifest = pd.DataFrame(
        [
            {
                "mode": "PPL",
                "class_id": "class17",
                "split": "train",
                "sample_id": "train",
                "local_relative_path": train_path.as_posix(),
            },
            {
                "mode": "PPL",
                "class_id": "class17",
                "split": "val",
                "sample_id": "val",
                "local_relative_path": val_path.as_posix(),
            },
        ]
    )

    report = find_cross_split_near_duplicates(
        manifest,
        project_root=tmp_path,
        max_hamming_distance=0,
    )

    assert len(report) == 1
    assert int(report.loc[0, "phash_hamming_distance"]) == 0


def test_applies_manual_near_duplicate_review() -> None:
    pair = {
        "mode": "XPL",
        "class_id": "class17",
        "split_a": "test",
        "sample_id_a": "10",
        "split_b": "train",
        "sample_id_b": "Oolitic.135",
    }
    report = pd.DataFrame([{**pair, "phash_hamming_distance": 4}])
    reviews = pd.DataFrame(
        [
            {
                **pair,
                "manual_decision": "exclude_train_near_duplicate",
                "review_notes": "Mesmo campo petrográfico com brilho diferente.",
            }
        ]
    )

    reviewed = apply_near_duplicate_reviews(report, reviews)

    assert reviewed.loc[0, "manual_decision"] == "exclude_train_near_duplicate"
