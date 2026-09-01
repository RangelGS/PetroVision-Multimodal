from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from petrovision.embeddings import (
    l2_normalize_embeddings,
    load_embedding_bundle,
    save_embedding_bundle,
    select_model_inputs,
)


def test_select_model_inputs_keeps_only_approved_in_stable_order() -> None:
    report = pd.DataFrame(
        [
            {
                "sample_id": "b",
                "mode": "XPL",
                "split": "test",
                "class_name": "class22_pore",
                "path": "b.jpg",
                "use_for_model": "False",
            },
            {
                "sample_id": "a",
                "mode": "PPL",
                "split": "train",
                "class_name": "class10_cemented_fracture",
                "path": "a.jpg",
                "use_for_model": "True",
            },
            {
                "sample_id": "c",
                "mode": "XPL",
                "split": "val",
                "class_name": "class13_micritic_limestone",
                "path": "c.jpg",
                "use_for_model": "1",
            },
        ]
    )

    selected = select_model_inputs(report)

    assert selected["sample_id"].tolist() == ["a", "c"]
    assert selected["embedding_row"].tolist() == [0, 1]


def test_select_model_inputs_rejects_identity_reused_across_splits() -> None:
    report = pd.DataFrame(
        [
            {
                "sample_id": "Oolitic.20",
                "mode": "PPL",
                "split": split,
                "class_name": "class17_oolite",
                "path": f"{split}.jpg",
                "use_for_model": True,
            }
            for split in ("train", "val")
        ]
    )

    with pytest.raises(ValueError, match="reutilizados entre treino"):
        select_model_inputs(report)


def test_l2_normalize_and_bundle_round_trip(tmp_path: Path) -> None:
    matrix = l2_normalize_embeddings(np.array([[3.0, 4.0], [0.0, 2.0]]))
    index = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "sample_id": ["a", "b"],
            "mode": ["PPL", "XPL"],
            "split": ["train", "test"],
            "class_name": ["class10", "class22"],
            "path": ["a.jpg", "b.jpg"],
        }
    )
    npz_path = tmp_path / "embeddings.npz"
    csv_path = tmp_path / "index.csv"

    save_embedding_bundle(matrix, index, npz_path, csv_path)
    loaded_matrix, loaded_index = load_embedding_bundle(npz_path, csv_path)

    np.testing.assert_allclose(np.linalg.norm(loaded_matrix, axis=1), 1.0)
    np.testing.assert_allclose(loaded_matrix, matrix)
    assert loaded_index["sample_id"].tolist() == ["a", "b"]


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="vetores nulos"):
        l2_normalize_embeddings(np.array([[0.0, 0.0]], dtype=np.float32))
