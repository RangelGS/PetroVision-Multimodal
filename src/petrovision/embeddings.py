"""Contratos e persistência para embeddings visuais reproduzíveis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from petrovision.catalog import validate_split_disjointness


EMBEDDING_KEY = "embeddings"
IDENTITY_COLUMNS = ["sample_id", "mode", "split", "class_name", "path"]


def _coerce_boolean(series: pd.Series, column: str) -> pd.Series:
    """Converte uma coluna booleana de CSV sem aceitar valores ambíguos."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"Valores booleanos inválidos em {column}: {invalid}")
    return normalized.map(mapping).astype(bool)


def select_model_inputs(quality_report: pd.DataFrame) -> pd.DataFrame:
    """Seleciona, ordena e valida as imagens aprovadas para o modelo."""

    required = set(IDENTITY_COLUMNS + ["use_for_model"])
    missing = required.difference(quality_report.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no relatório de qualidade: {sorted(missing)}")

    accepted_mask = _coerce_boolean(quality_report["use_for_model"], "use_for_model")
    selected = quality_report.loc[accepted_mask, IDENTITY_COLUMNS].copy()
    if selected.empty:
        raise ValueError("Nenhuma imagem foi aprovada para extração de embeddings.")

    duplicated = selected.duplicated(subset=IDENTITY_COLUMNS[:-1], keep=False)
    if duplicated.any():
        rows = selected.loc[duplicated, IDENTITY_COLUMNS[:-1]]
        raise ValueError(
            "Há identidades repetidas entre as imagens aprovadas:\n"
            f"{rows.to_string(index=False)}"
        )

    validate_split_disjointness(selected)

    split_order = pd.CategoricalDtype(["train", "val", "test"], ordered=True)
    selected["split"] = selected["split"].astype(split_order)
    selected = selected.sort_values(
        ["split", "mode", "class_name", "sample_id"]
    ).reset_index(drop=True)
    selected["split"] = selected["split"].astype("object")
    selected.insert(0, "embedding_row", np.arange(len(selected), dtype=int))
    return selected


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normaliza cada vetor e rejeita linhas nulas ou não finitas."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("A matriz de embeddings deve ter duas dimensões.")
    if not np.isfinite(matrix).all():
        raise ValueError("A matriz de embeddings contém valores não finitos.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A matriz de embeddings contém vetores nulos.")
    return matrix / norms


def validate_embedding_alignment(
    embeddings: np.ndarray,
    index: pd.DataFrame,
) -> None:
    """Garante alinhamento 1:1 entre linhas da matriz e do índice."""

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("A matriz de embeddings deve ter duas dimensões.")
    if len(matrix) != len(index):
        raise ValueError(
            f"Matriz e índice têm tamanhos diferentes: {len(matrix)} != {len(index)}"
        )
    if "embedding_row" not in index:
        raise ValueError("O índice precisa conter a coluna 'embedding_row'.")
    expected = np.arange(len(index), dtype=int)
    actual = index["embedding_row"].to_numpy(dtype=int)
    if not np.array_equal(actual, expected):
        raise ValueError("A coluna embedding_row não corresponde à ordem da matriz.")
    if not np.isfinite(matrix).all():
        raise ValueError("A matriz de embeddings contém valores não finitos.")


def save_embedding_bundle(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    embeddings_path: str | Path,
    index_path: str | Path,
) -> None:
    """Salva matriz compactada e índice CSV após validar o alinhamento."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    validate_embedding_alignment(matrix, index)
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, **{EMBEDDING_KEY: matrix})
    index.to_csv(index_path, index=False)


def load_embedding_bundle(
    embeddings_path: str | Path,
    index_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Carrega um artefato sem permitir objetos serializados por pickle."""

    with np.load(Path(embeddings_path), allow_pickle=False) as archive:
        if EMBEDDING_KEY not in archive:
            raise ValueError(f"Chave ausente no NPZ: {EMBEDDING_KEY}")
        embeddings = np.asarray(archive[EMBEDDING_KEY], dtype=np.float32)
    index = pd.read_csv(index_path)
    validate_embedding_alignment(embeddings, index)
    return embeddings, index


def save_run_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """Persiste metadados técnicos legíveis do processo de extração."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
