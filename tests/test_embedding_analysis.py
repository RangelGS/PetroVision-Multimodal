import numpy as np
import pandas as pd

from petrovision.embedding_analysis import (
    clustering_diagnostics,
    linear_probe_suite,
    prototype_similarity,
)


def synthetic_multimodal_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    classes = ["class10", "class13", "class17", "class22"]
    splits = {"train": 3, "val": 2, "test": 2}
    for mode_index, mode in enumerate(["PPL", "XPL"]):
        for class_index, class_name in enumerate(classes):
            center = np.zeros(8, dtype=np.float32)
            center[class_index] = 4.0
            center[4 + mode_index] = 0.15
            for split, count in splits.items():
                for sample_number in range(count):
                    vector = center + rng.normal(0, 0.03, size=8)
                    vectors.append(vector.astype(np.float32))
                    rows.append(
                        {
                            "embedding_row": len(rows),
                            "sample_id": f"{mode}-{class_name}-{split}-{sample_number}",
                            "mode": mode,
                            "split": split,
                            "class_name": class_name,
                            "path": "image.jpg",
                        }
                    )
    matrix = np.stack(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, pd.DataFrame(rows)


def test_linear_probe_suite_preserves_test_for_final_evaluation() -> None:
    embeddings, index = synthetic_multimodal_embeddings()

    metrics, predictions = linear_probe_suite(
        embeddings, index, c_values=[0.1, 1.0, 10.0]
    )

    assert len(metrics) == 6
    assert set(metrics["relation"]) == {"within_domain", "cross_domain", "combined"}
    assert (metrics["macro_f1"] == 1.0).all()
    assert set(predictions["sample_id"]).issubset(
        set(index.loc[index["split"].eq("test"), "sample_id"])
    )


def test_prototypes_and_clusters_reflect_synthetic_classes() -> None:
    embeddings, index = synthetic_multimodal_embeddings()

    similarities, summary = prototype_similarity(embeddings, index)
    cluster_metrics, clusters = clustering_diagnostics(
        embeddings, index, n_clusters=4
    )

    assert len(similarities) == 16
    assert summary["prototype_alignment_gap"] > 0.9
    assert cluster_metrics.loc[0, "adjusted_rand_class"] == 1.0
    assert len(clusters) == len(index)
