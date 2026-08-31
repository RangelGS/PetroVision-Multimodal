"""Executa o controle de qualidade e salva relatórios CSV."""

from __future__ import annotations

import pandas as pd

from petrovision.config import load_config, project_path
from petrovision.quality import apply_manual_reviews, evaluate_catalog, summarise_quality


def main() -> int:
    config = load_config()
    catalog_path = project_path(config["paths"]["catalog"])
    if not catalog_path.exists():
        print("Catálogo não encontrado. Execute primeiro: python scripts/build_catalog.py")
        return 1

    catalog = pd.read_csv(catalog_path)
    automatic_report = evaluate_catalog(catalog, config["quality"])
    manual_review_path = project_path(config["paths"]["manual_quality_review"])
    if not manual_review_path.exists():
        print(f"Revisão manual não encontrada: {manual_review_path}")
        return 1
    manual_reviews = pd.read_csv(manual_review_path)
    report = apply_manual_reviews(automatic_report, manual_reviews)
    summary = summarise_quality(report)

    report_path = project_path(config["paths"]["quality_report"])
    summary_path = project_path(config["paths"]["quality_summary"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Relatório detalhado: {report_path}")
    print(f"Resumo: {summary_path}")
    print(summary.to_string(index=False))
    print("\nObservação: alertas automáticos indicam revisão, não exclusão.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
