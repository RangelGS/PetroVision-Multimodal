"""Analisa embeddings DINOv2 sem ajustar o modelo fundacional."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from petrovision.config import load_config, project_path
from petrovision.embedding_analysis import (
    clustering_diagnostics,
    linear_probe_suite,
    prototype_similarity,
)
from petrovision.embeddings import load_embedding_bundle


def save_pca_plot(coordinates: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=coordinates,
        x="pca_1",
        y="pca_2",
        hue="class_name",
        style="mode",
        s=85,
        alpha=0.82,
        ax=axis,
    )
    axis.set_title("DINOv2 — projeção PCA por classe e modalidade")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_prototype_plot(similarity: pd.DataFrame, output: Path) -> None:
    matrix = similarity.pivot(
        index="source_class", columns="target_class", values="cosine_similarity"
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=axis)
    axis.set_title("Similaridade entre protótipos PPL e XPL (treino)")
    axis.set_xlabel("Classe-alvo XPL")
    axis.set_ylabel("Classe-fonte PPL")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(predictions: pd.DataFrame, output: Path) -> None:
    classes = sorted(predictions["true_class"].unique())
    domains = sorted(predictions["training_domain"].unique())
    modes = sorted(predictions["mode"].unique())
    figure, axes = plt.subplots(len(domains), len(modes), figsize=(15, 18))
    for row, domain in enumerate(domains):
        for column, mode in enumerate(modes):
            subset = predictions[
                predictions["training_domain"].eq(domain)
                & predictions["mode"].eq(mode)
            ]
            matrix = confusion_matrix(
                subset["true_class"], subset["predicted_class"], labels=classes
            )
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=classes,
                yticklabels=classes,
                ax=axes[row, column],
            )
            axes[row, column].set_title(f"Treino {domain} → teste {mode}")
            axes[row, column].set_xlabel("Predita")
            axes[row, column].set_ylabel("Real")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    config = load_config()
    embeddings_path = project_path(config["paths"]["dinov2_embeddings"])
    index_path = project_path(config["paths"]["dinov2_index"])
    if not embeddings_path.exists() or not index_path.exists():
        print("Embeddings ou índice não encontrados.")
        print("Execute primeiro: python scripts/extract_dinov2_embeddings.py")
        return 1

    embeddings, index = load_embedding_bundle(embeddings_path, index_path)
    output_dir = project_path(config["paths"]["dinov2_analysis_dir"])
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_path = project_path(config["paths"]["dinov2_run"])
    shutil.copy2(index_path, table_dir / "dinov2_embedding_index.csv")
    if run_path.exists():
        shutil.copy2(run_path, table_dir / "dinov2_embedding_run.json")

    metrics, predictions = linear_probe_suite(
        embeddings,
        index,
        c_values=config["analysis"]["linear_probe_c_values"],
        random_seed=int(config["project"]["random_seed"]),
    )
    similarity, prototype_summary = prototype_similarity(embeddings, index)
    cluster_metrics, clusters = clustering_diagnostics(
        embeddings,
        index,
        n_clusters=int(config["analysis"]["kmeans_clusters"]),
        random_seed=int(config["project"]["random_seed"]),
    )

    train_mask = index["split"].eq("train").to_numpy()
    pca = PCA(n_components=2, random_state=int(config["project"]["random_seed"]))
    pca.fit(embeddings[train_mask])
    pca_values = pca.transform(embeddings)
    coordinates = index.copy()
    coordinates["pca_1"] = pca_values[:, 0]
    coordinates["pca_2"] = pca_values[:, 1]
    coordinates["kmeans_cluster"] = clusters

    metrics.to_csv(table_dir / "linear_probe_metrics.csv", index=False)
    predictions.to_csv(table_dir / "linear_probe_predictions.csv", index=False)
    similarity.to_csv(table_dir / "prototype_similarity.csv", index=False)
    cluster_metrics.to_csv(table_dir / "clustering_metrics.csv", index=False)
    coordinates.to_csv(table_dir / "pca_coordinates.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "samples": len(index),
                "embedding_dimensions": embeddings.shape[1],
                "pca_explained_variance_2d": float(
                    pca.explained_variance_ratio_.sum()
                ),
                **prototype_summary,
            }
        ]
    )
    summary.to_csv(table_dir / "embedding_summary.csv", index=False)

    save_pca_plot(coordinates, figure_dir / "pca_class_mode.png")
    save_prototype_plot(similarity, figure_dir / "prototype_similarity.png")
    save_confusion_plot(predictions, figure_dir / "linear_probe_confusions.png")

    report = output_dir / "DINOV2_REPORT.md"
    report.write_text(
        "# Relatório automático — DINOv2\n\n"
        "## Classificadores lineares\n\n"
        f"{metrics.to_markdown(index=False, floatfmt='.3f')}\n\n"
        "## Diagnóstico de agrupamento\n\n"
        f"{cluster_metrics.to_markdown(index=False, floatfmt='.3f')}\n\n"
        "## Alinhamento de protótipos\n\n"
        f"{summary.to_markdown(index=False, floatfmt='.3f')}\n\n"
        "> Resultados exploratórios em um subconjunto pequeno. O teste foi mantido "
        "fora da seleção de hiperparâmetros. PPL e XPL são domínios não pareados.\n",
        encoding="utf-8",
    )

    print("Análise concluída.")
    print(f"Relatório: {report}")
    print("\nClassificadores lineares:")
    print(metrics.to_string(index=False))
    print("\nAgrupamento:")
    print(cluster_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
