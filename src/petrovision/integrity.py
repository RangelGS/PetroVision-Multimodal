"""Auditorias de integridade visual entre divisões do conjunto de dados."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


NEAR_DUPLICATE_COLUMNS = [
    "mode",
    "class_id",
    "split_a",
    "sample_id_a",
    "path_a",
    "split_b",
    "sample_id_b",
    "path_b",
    "phash_hamming_distance",
]

PAIR_ID_COLUMNS = [
    "mode",
    "class_id",
    "split_a",
    "sample_id_a",
    "split_b",
    "sample_id_b",
]


def perceptual_hash(path: str | Path, *, hash_size: int = 8) -> int:
    """Calcula pHash de 64 bits, resistente a pequenas mudanças de codificação."""

    if hash_size < 2:
        raise ValueError("hash_size deve ser pelo menos 2.")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem para pHash: {path}")

    dct_size = hash_size * 4
    resized = cv2.resize(image, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
    frequencies = cv2.dct(resized.astype(np.float32))[:hash_size, :hash_size]
    flattened = frequencies.ravel()
    threshold = float(np.median(flattened[1:]))
    bits = flattened > threshold
    bits[0] = False

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_hamming_distance(first: int, second: int) -> int:
    """Retorna quantos bits diferem entre duas assinaturas perceptuais."""

    return int(first ^ second).bit_count()


def find_cross_split_near_duplicates(
    manifest: pd.DataFrame,
    *,
    project_root: str | Path,
    max_hamming_distance: int = 4,
) -> pd.DataFrame:
    """Lista candidatos visualmente próximos entre divisões da mesma classe.

    O resultado é uma triagem conservadora. Uma coincidência de pHash deve ser
    revisada visualmente e não é excluída automaticamente.
    """

    required = {
        "mode",
        "class_id",
        "split",
        "sample_id",
        "local_relative_path",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes para a auditoria perceptual: {sorted(missing)}"
        )
    if max_hamming_distance < 0:
        raise ValueError("max_hamming_distance não pode ser negativo.")

    root = Path(project_root)
    audited = manifest.copy().reset_index(drop=True)
    audited["_phash"] = [
        perceptual_hash(root / relative_path)
        for relative_path in audited["local_relative_path"].astype(str)
    ]

    candidates: list[dict[str, object]] = []
    for (mode, class_id), group in audited.groupby(["mode", "class_id"]):
        records = group.to_dict(orient="records")
        for first, second in combinations(records, 2):
            if str(first["split"]) == str(second["split"]):
                continue
            distance = phash_hamming_distance(first["_phash"], second["_phash"])
            if distance > max_hamming_distance:
                continue
            candidates.append(
                {
                    "mode": mode,
                    "class_id": class_id,
                    "split_a": first["split"],
                    "sample_id_a": first["sample_id"],
                    "path_a": first["local_relative_path"],
                    "split_b": second["split"],
                    "sample_id_b": second["sample_id"],
                    "path_b": second["local_relative_path"],
                    "phash_hamming_distance": distance,
                }
            )

    report = pd.DataFrame.from_records(candidates, columns=NEAR_DUPLICATE_COLUMNS)
    if report.empty:
        return report
    return report.sort_values(
        ["phash_hamming_distance", "mode", "class_id", "split_a", "sample_id_a"]
    ).reset_index(drop=True)


def apply_near_duplicate_reviews(
    report: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Anexa decisões visuais auditáveis aos candidatos encontrados."""

    review_columns = PAIR_ID_COLUMNS + ["manual_decision", "review_notes"]
    missing_report = set(PAIR_ID_COLUMNS).difference(report.columns)
    missing_reviews = set(review_columns).difference(reviews.columns)
    if missing_report:
        raise ValueError(f"Colunas ausentes no relatório pHash: {sorted(missing_report)}")
    if missing_reviews:
        raise ValueError(
            f"Colunas ausentes na revisão pHash: {sorted(missing_reviews)}"
        )

    duplicated = reviews.duplicated(subset=PAIR_ID_COLUMNS, keep=False)
    if duplicated.any():
        raise ValueError("A revisão pHash contém decisões duplicadas para um par.")

    allowed = {"distinct_keep", "exclude_train_near_duplicate"}
    decisions = set(reviews["manual_decision"].dropna().astype(str))
    invalid = decisions.difference(allowed)
    if invalid:
        raise ValueError(f"Decisões pHash inválidas: {sorted(invalid)}")

    merged = report.merge(
        reviews[review_columns],
        on=PAIR_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged["manual_decision"] = merged["manual_decision"].fillna("pending_review")
    merged["review_notes"] = merged["review_notes"].fillna("")
    return merged
