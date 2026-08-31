"""Análises reproduzíveis sobre embeddings congelados do DINOv2."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _require_columns(index: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(index.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no índice de embeddings: {sorted(missing)}")


def _require_all_classes(labels: np.ndarray, expected: list[str], context: str) -> None:
    present = sorted(set(labels.astype(str)))
    if present != expected:
        raise ValueError(
            f"Classes incompletas em {context}: encontradas {present}; esperadas {expected}"
        )


def linear_probe_suite(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    c_values: Iterable[float],
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seleciona C na validação e avalia probes intra e cross-domain no teste."""

    _require_columns(index, ["sample_id", "mode", "split", "class_name"])
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) != len(index):
        raise ValueError("Embeddings e índice precisam ter o mesmo número de linhas.")

    classes = sorted(index["class_name"].astype(str).unique())
    modes = sorted(index["mode"].astype(str).unique())
    if len(modes) != 2:
        raise ValueError(f"A análise espera duas modalidades; encontradas: {modes}")

    c_grid = [float(value) for value in c_values]
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("Os valores de C devem ser positivos.")

    experiments: list[tuple[str, list[str]]] = [
        (modes[0], [modes[0]]),
        (modes[1], [modes[1]]),
        ("+".join(modes), modes),
    ]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for training_domain, train_modes in experiments:
        train_mask = index["split"].eq("train") & index["mode"].isin(train_modes)
        val_mask = index["split"].eq("val") & index["mode"].isin(train_modes)
        fit_mask = index["split"].isin(["train", "val"]) & index["mode"].isin(
            train_modes
        )

        y_train = index.loc[train_mask, "class_name"].astype(str).to_numpy()
        y_val = index.loc[val_mask, "class_name"].astype(str).to_numpy()
        _require_all_classes(y_train, classes, f"treino {training_domain}")
        _require_all_classes(y_val, classes, f"validação {training_domain}")

        validation_scores: list[tuple[float, float]] = []
        for c_value in c_grid:
            candidate = LogisticRegression(
                C=c_value,
                max_iter=5000,
                solver="lbfgs",
                random_state=random_seed,
            )
            candidate.fit(matrix[train_mask.to_numpy()], y_train)
            predicted_val = candidate.predict(matrix[val_mask.to_numpy()])
            score = f1_score(y_val, predicted_val, average="macro", zero_division=0)
            validation_scores.append((score, c_value))

        best_score = max(score for score, _ in validation_scores)
        best_c = min(c for score, c in validation_scores if score == best_score)
        final_model = LogisticRegression(
            C=best_c,
            max_iter=5000,
            solver="lbfgs",
            random_state=random_seed,
        )
        final_model.fit(
            matrix[fit_mask.to_numpy()],
            index.loc[fit_mask, "class_name"].astype(str).to_numpy(),
        )

        for evaluation_mode in modes:
            test_mask = index["split"].eq("test") & index["mode"].eq(evaluation_mode)
            y_test = index.loc[test_mask, "class_name"].astype(str).to_numpy()
            _require_all_classes(y_test, classes, f"teste {evaluation_mode}")
            predicted = final_model.predict(matrix[test_mask.to_numpy()])
            relation = (
                "combined"
                if len(train_modes) == len(modes)
                else ("within_domain" if evaluation_mode in train_modes else "cross_domain")
            )
            metric_rows.append(
                {
                    "training_domain": training_domain,
                    "evaluation_mode": evaluation_mode,
                    "relation": relation,
                    "selected_c": best_c,
                    "validation_macro_f1": best_score,
                    "accuracy": accuracy_score(y_test, predicted),
                    "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                    "macro_f1": f1_score(
                        y_test, predicted, average="macro", zero_division=0
                    ),
                    "fit_samples": int(fit_mask.sum()),
                    "test_samples": int(test_mask.sum()),
                }
            )
            test_rows = index.loc[
                test_mask, ["embedding_row", "sample_id", "mode", "class_name"]
            ].copy()
            test_rows["training_domain"] = training_domain
            test_rows["true_class"] = y_test
            test_rows["predicted_class"] = predicted
            prediction_rows.extend(test_rows.to_dict(orient="records"))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["training_domain", "evaluation_mode"]
    ).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["training_domain", "mode", "sample_id"]
    ).reset_index(drop=True)
    return metrics, predictions


def prototype_similarity(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    split: str = "train",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compara protótipos de classe entre as duas modalidades por cosseno."""

    _require_columns(index, ["mode", "split", "class_name"])
    modes = sorted(index["mode"].astype(str).unique())
    classes = sorted(index["class_name"].astype(str).unique())
    if len(modes) != 2:
        raise ValueError("A matriz de protótipos exige exatamente duas modalidades.")

    prototypes: dict[tuple[str, str], np.ndarray] = {}
    for mode in modes:
        for class_name in classes:
            mask = (
                index["split"].eq(split)
                & index["mode"].eq(mode)
                & index["class_name"].eq(class_name)
            )
            if not mask.any():
                raise ValueError(f"Sem amostras para {mode}/{class_name}/{split}.")
            vector = np.asarray(embeddings)[mask.to_numpy()].mean(axis=0)
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError(f"Protótipo nulo para {mode}/{class_name}/{split}.")
            prototypes[(mode, class_name)] = vector / norm

    rows: list[dict[str, object]] = []
    values = np.empty((len(classes), len(classes)), dtype=float)
    for row_idx, ppl_class in enumerate(classes):
        for col_idx, xpl_class in enumerate(classes):
            similarity = float(
                np.dot(
                    prototypes[(modes[0], ppl_class)],
                    prototypes[(modes[1], xpl_class)],
                )
            )
            values[row_idx, col_idx] = similarity
            rows.append(
                {
                    "source_mode": modes[0],
                    "source_class": ppl_class,
                    "target_mode": modes[1],
                    "target_class": xpl_class,
                    "cosine_similarity": similarity,
                    "same_class": ppl_class == xpl_class,
                }
            )

    diagonal = np.diag(values)
    off_diagonal = values[~np.eye(len(classes), dtype=bool)]
    summary = {
        "same_class_mean_cosine": float(diagonal.mean()),
        "different_class_mean_cosine": float(off_diagonal.mean()),
        "prototype_alignment_gap": float(diagonal.mean() - off_diagonal.mean()),
    }
    return pd.DataFrame(rows), summary


def clustering_diagnostics(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mede se agrupamentos refletem mais a classe ou a modalidade óptica."""

    _require_columns(index, ["mode", "class_name"])
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) != len(index):
        raise ValueError("Embeddings e índice precisam ter o mesmo número de linhas.")
    model = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_seed)
    clusters = model.fit_predict(matrix)
    metrics = pd.DataFrame(
        [
            {
                "samples": len(index),
                "clusters": n_clusters,
                "adjusted_rand_class": adjusted_rand_score(index["class_name"], clusters),
                "nmi_class": normalized_mutual_info_score(index["class_name"], clusters),
                "adjusted_rand_mode": adjusted_rand_score(index["mode"], clusters),
                "nmi_mode": normalized_mutual_info_score(index["mode"], clusters),
                "silhouette_clusters": silhouette_score(matrix, clusters),
            }
        ]
    )
    return metrics, clusters
