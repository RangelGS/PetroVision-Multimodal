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
    """Aplica limites transparentes para triagem, sem excluir imagens."""

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

    automatic_ok = not reasons
    return {
        "automatic_quality_ok": automatic_ok,
        "review_required": not automatic_ok,
        "automatic_status": "automatic_pass" if automatic_ok else "review_required",
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
                    "automatic_quality_ok": False,
                    "review_required": True,
                    "automatic_status": "read_error",
                    "quality_flags": "erro_leitura",
                    "read_error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def apply_manual_reviews(
    report: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Combina a triagem automática com decisões visuais documentadas."""

    keys = ["mode", "split", "class_name", "sample_id"]
    missing_report = set(keys + ["automatic_quality_ok", "read_error"]).difference(
        report.columns
    )
    if missing_report:
        raise ValueError(f"Colunas ausentes no relatório: {sorted(missing_report)}")

    review_columns = keys + ["manual_decision", "review_notes"]
    missing_reviews = set(review_columns).difference(reviews.columns)
    if missing_reviews:
        raise ValueError(f"Colunas ausentes na revisão manual: {sorted(missing_reviews)}")

    duplicated = reviews.duplicated(subset=keys, keep=False)
    if duplicated.any():
        raise ValueError("A revisão manual contém decisões duplicadas para uma imagem.")

    allowed = {"keep", "exclude"}
    decisions = set(reviews["manual_decision"].dropna().astype(str))
    invalid = decisions.difference(allowed)
    if invalid:
        raise ValueError(f"Decisões manuais inválidas: {sorted(invalid)}")

    merged = report.merge(reviews[review_columns], on=keys, how="left", validate="one_to_one")
    merged["manual_decision"] = merged["manual_decision"].fillna("")
    merged["review_notes"] = merged["review_notes"].fillna("")

    has_read_error = merged["read_error"].fillna("").astype(str).ne("")
    automatic_ok = merged["automatic_quality_ok"].astype(bool)
    manually_kept = merged["manual_decision"].eq("keep")
    manually_excluded = merged["manual_decision"].eq("exclude")

    merged["final_status"] = "pending_review"
    merged.loc[automatic_ok, "final_status"] = "automatic_pass"
    merged.loc[manually_kept, "final_status"] = "manual_keep"
    merged.loc[manually_excluded, "final_status"] = "manual_exclude"
    merged.loc[has_read_error, "final_status"] = "read_error"
    merged["use_for_model"] = (automatic_ok | manually_kept) & ~(
        manually_excluded | has_read_error
    )
    return merged


def summarise_quality(report: pd.DataFrame) -> pd.DataFrame:
    """Resume triagem, revisão manual e métricas por modalidade e divisão."""

    if report.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "split",
                "images",
                "automatic_pass",
                "manual_keep",
                "pending_review",
                "accepted_for_model",
                "acceptance_rate",
            ]
        )

    summary = (
        report.groupby(["mode", "split"], dropna=False)
        .agg(
            images=("path", "count"),
            automatic_pass=("automatic_quality_ok", "sum"),
            manual_keep=("final_status", lambda values: (values == "manual_keep").sum()),
            pending_review=(
                "final_status",
                lambda values: (values == "pending_review").sum(),
            ),
            accepted_for_model=("use_for_model", "sum"),
            mean_brightness=("brightness", "mean"),
            mean_contrast=("contrast", "mean"),
            mean_sharpness=("sharpness", "mean"),
        )
        .reset_index()
    )
    summary["acceptance_rate"] = summary["accepted_for_model"] / summary["images"]
    return summary
