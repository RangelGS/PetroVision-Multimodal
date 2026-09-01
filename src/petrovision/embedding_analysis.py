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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


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


def _normalize_c_grid(c_values: Iterable[float]) -> list[float]:
    grid = [float(value) for value in c_values]
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("Os valores de C devem ser positivos.")
    return grid


def _select_c_with_inner_cv(
    matrix: np.ndarray,
    labels: np.ndarray,
    c_grid: list[float],
    inner_splits: int,
    random_seed: int,
) -> tuple[float, float]:
    """Seleciona C por macro-F1 sem observar a dobra externa."""

    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_seed,
    )
    candidate_scores: list[tuple[float, float]] = []
    for c_value in c_grid:
        fold_scores: list[float] = []
        for train_rows, validation_rows in splitter.split(matrix, labels):
            candidate = LogisticRegression(
                C=c_value,
                max_iter=5000,
                solver="lbfgs",
                random_state=random_seed,
            )
            candidate.fit(matrix[train_rows], labels[train_rows])
            predicted = candidate.predict(matrix[validation_rows])
            fold_scores.append(
                f1_score(
                    labels[validation_rows],
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            )
        candidate_scores.append((float(np.mean(fold_scores)), c_value))

    best_score = max(score for score, _ in candidate_scores)
    best_c = min(c for score, c in candidate_scores if score == best_score)
    return best_c, best_score


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

    c_grid = _normalize_c_grid(c_values)

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


def repeated_probe_stability(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    c_values: Iterable[float],
    outer_splits: int = 5,
    outer_repeats: int = 5,
    inner_splits: int = 3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mede estabilidade com validação repetida aninhada em treino+validação.

    As linhas do teste oficial são deliberadamente ignoradas. Em cada dobra
    externa, o valor de C é escolhido apenas por validação cruzada interna nas
    linhas de ajuste daquela dobra.
    """

    _require_columns(index, ["sample_id", "mode", "split", "class_name"])
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) != len(index):
        raise ValueError("Embeddings e índice precisam ter o mesmo número de linhas.")
    if outer_splits < 2:
        raise ValueError("outer_splits deve ser pelo menos 2.")
    if outer_repeats < 1:
        raise ValueError("outer_repeats deve ser pelo menos 1.")
    if inner_splits < 2:
        raise ValueError("inner_splits deve ser pelo menos 2.")

    modes = sorted(index["mode"].astype(str).unique())
    if len(modes) != 2:
        raise ValueError(f"A análise espera duas modalidades; encontradas: {modes}")
    development_mask = index["split"].isin(["train", "val"])
    development_classes = sorted(
        index.loc[development_mask, "class_name"].astype(str).unique()
    )
    c_grid = _normalize_c_grid(c_values)

    mode_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for mode_number, mode in enumerate(modes):
        positions = np.flatnonzero(
            (development_mask & index["mode"].eq(mode)).to_numpy()
        )
        labels = index.iloc[positions]["class_name"].astype(str).to_numpy()
        _require_all_classes(labels, development_classes, f"desenvolvimento {mode}")
        class_counts = pd.Series(labels).value_counts()
        if int(class_counts.min()) < outer_splits:
            raise ValueError(
                f"Cada classe de {mode} precisa de ao menos {outer_splits} amostras "
                "para as divisões externas."
            )
        largest_outer_fold = int(np.ceil(class_counts.max() / outer_splits))
        smallest_outer_train = int(class_counts.min()) - largest_outer_fold
        if smallest_outer_train < inner_splits:
            raise ValueError(
                f"As dobras externas de {mode} deixam menos de {inner_splits} "
                "amostras por classe para a validação interna."
            )

        splitter = RepeatedStratifiedKFold(
            n_splits=outer_splits,
            n_repeats=outer_repeats,
            random_state=random_seed + (mode_number * 10_000),
        )
        mode_splits[mode] = [
            (positions[fit_rows], positions[evaluation_rows])
            for fit_rows, evaluation_rows in splitter.split(matrix[positions], labels)
        ]

    experiments: list[tuple[str, list[str]]] = [
        (modes[0], [modes[0]]),
        (modes[1], [modes[1]]),
        ("+".join(modes), modes),
    ]
    metric_rows: list[dict[str, object]] = []
    total_outer_runs = outer_splits * outer_repeats
    for run_number in range(total_outer_runs):
        repeat_number = (run_number // outer_splits) + 1
        fold_number = (run_number % outer_splits) + 1
        for experiment_number, (training_domain, train_modes) in enumerate(experiments):
            fit_positions = np.concatenate(
                [mode_splits[mode][run_number][0] for mode in train_modes]
            )
            fit_labels = (
                index.iloc[fit_positions]["class_name"].astype(str).to_numpy()
            )
            _require_all_classes(
                fit_labels,
                development_classes,
                f"ajuste repetido {training_domain}",
            )
            selection_seed = (
                random_seed + (run_number * 101) + (experiment_number * 10_000)
            )
            selected_c, inner_score = _select_c_with_inner_cv(
                matrix[fit_positions],
                fit_labels,
                c_grid,
                inner_splits=inner_splits,
                random_seed=selection_seed,
            )
            final_model = LogisticRegression(
                C=selected_c,
                max_iter=5000,
                solver="lbfgs",
                random_state=selection_seed,
            )
            final_model.fit(matrix[fit_positions], fit_labels)

            for evaluation_mode in modes:
                evaluation_positions = mode_splits[evaluation_mode][run_number][1]
                evaluation_labels = (
                    index.iloc[evaluation_positions]["class_name"]
                    .astype(str)
                    .to_numpy()
                )
                _require_all_classes(
                    evaluation_labels,
                    development_classes,
                    f"avaliação repetida {evaluation_mode}",
                )
                predicted = final_model.predict(matrix[evaluation_positions])
                relation = (
                    "combined"
                    if len(train_modes) == len(modes)
                    else (
                        "within_domain"
                        if evaluation_mode in train_modes
                        else "cross_domain"
                    )
                )
                metric_rows.append(
                    {
                        "outer_repeat": repeat_number,
                        "outer_fold": fold_number,
                        "training_domain": training_domain,
                        "evaluation_mode": evaluation_mode,
                        "relation": relation,
                        "selected_c": selected_c,
                        "inner_validation_macro_f1": inner_score,
                        "accuracy": accuracy_score(evaluation_labels, predicted),
                        "balanced_accuracy": balanced_accuracy_score(
                            evaluation_labels, predicted
                        ),
                        "macro_f1": f1_score(
                            evaluation_labels,
                            predicted,
                            average="macro",
                            zero_division=0,
                        ),
                        "fit_samples": len(fit_positions),
                        "evaluation_samples": len(evaluation_positions),
                    }
                )

    folds = pd.DataFrame(metric_rows).sort_values(
        ["training_domain", "evaluation_mode", "outer_repeat", "outer_fold"]
    ).reset_index(drop=True)
    group_columns = ["training_domain", "evaluation_mode", "relation"]
    summary = (
        folds.groupby(group_columns, sort=True)
        .agg(
            stability_runs=("macro_f1", "size"),
            selected_c_median=("selected_c", "median"),
            inner_validation_macro_f1_mean=(
                "inner_validation_macro_f1",
                "mean",
            ),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
        )
        .reset_index()
    )
    return folds, summary


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
