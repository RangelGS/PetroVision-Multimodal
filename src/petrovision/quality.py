"""Métricas simples e reprodutíveis de qualidade de imagem com OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pandas as pd


def calculate_quality_metrics(path: str | Path) -> dict[str, float | int]:
    """Calcula brilho, contraste, nitidez e dimensões de uma imagem."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    return {
        "width": int(width),
        "height": int(height),
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def apply_quality_rules(
    metrics: Mapping[str, float | int],
    thresholds: Mapping[str, float],
) -> dict[str, bool | str]:
    """Aplica limites transparentes e retorna as causas de reprovação."""

    reasons: list[str] = []
    brightness = float(metrics["brightness"])
    contrast = float(metrics["contrast"])
    sharpness = float(metrics["sharpness"])

    if brightness < float(thresholds["min_brightness"]):
        reasons.append("baixa_luminosidade")
    if brightness > float(thresholds["max_brightness"]):
        reasons.append("alta_luminosidade")
    if contrast < float(thresholds["min_contrast"]):
        reasons.append("baixo_contraste")
    if sharpness < float(thresholds["min_sharpness"]):
        reasons.append("possivel_desfoque")

    return {
        "quality_ok": not reasons,
        "quality_flags": ";".join(reasons),
    }


def evaluate_catalog(
    catalog: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    """Avalia todas as imagens do catálogo sem alterar os arquivos originais."""

    if "path" not in catalog:
        raise ValueError("O catálogo precisa conter a coluna 'path'.")

    rows: list[dict[str, object]] = []
    for record in catalog.to_dict(orient="records"):
        try:
            metrics = calculate_quality_metrics(str(record["path"]))
            rules = apply_quality_rules(metrics, thresholds)
            rows.append({**record, **metrics, **rules, "read_error": ""})
        except (OSError, ValueError) as exc:
            rows.append(
                {
                    **record,
                    "width": pd.NA,
                    "height": pd.NA,
                    "brightness": np.nan,
                    "contrast": np.nan,
                    "sharpness": np.nan,
                    "quality_ok": False,
                    "quality_flags": "erro_leitura",
                    "read_error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def summarise_quality(report: pd.DataFrame) -> pd.DataFrame:
    """Resume aprovação e métricas por modalidade e divisão."""

    if report.empty:
        return pd.DataFrame(
            columns=["mode", "split", "images", "approved", "approval_rate"]
        )

    summary = (
        report.groupby(["mode", "split"], dropna=False)
        .agg(
            images=("path", "count"),
            approved=("quality_ok", "sum"),
            mean_brightness=("brightness", "mean"),
            mean_contrast=("contrast", "mean"),
            mean_sharpness=("sharpness", "mean"),
        )
        .reset_index()
    )
    summary["approval_rate"] = summary["approved"] / summary["images"]
    return summary

