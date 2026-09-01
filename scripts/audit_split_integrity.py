"""Audita identidades, hashes e possíveis duplicatas visuais entre splits."""

from __future__ import annotations

import argparse

import pandas as pd

from petrovision.config import PROJECT_ROOT, load_config, project_path
from petrovision.integrity import (
    apply_near_duplicate_reviews,
    find_cross_split_near_duplicates,
)
from petrovision.remote_data import (
    validate_disjoint_split_identities,
    validate_unique_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-phash-distance",
        type=int,
        default=4,
        help="Distância Hamming máxima do pHash para triagem (padrão: 4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    manifest_path = project_path(config["remote_dataset"]["manifest"])
    report_path = project_path(config["paths"]["near_duplicate_report"])
    review_path = project_path(config["paths"]["manual_near_duplicate_review"])
    manifest = pd.read_csv(manifest_path)
    rows = manifest.to_dict(orient="records")

    validate_unique_hashes(rows)
    validate_disjoint_split_identities(rows)
    report = find_cross_split_near_duplicates(
        manifest,
        project_root=PROJECT_ROOT,
        max_hamming_distance=args.max_phash_distance,
    )
    reviews = pd.read_csv(review_path)
    report = apply_near_duplicate_reviews(report, reviews)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("Auditoria de identidades entre splits: aprovada")
    print("Auditoria de hashes SHA-256: aprovada")
    pending = int(report["manual_decision"].eq("pending_review").sum())
    reviewed = len(report) - pending
    print(f"Candidatos perceptuais encontrados: {len(report)}")
    print(f"Candidatos já revisados: {reviewed}")
    print(f"Candidatos pendentes de revisão: {pending}")
    print(f"Relatório perceptual: {report_path}")
    if pending:
        print(
            "Observação: pHash é uma triagem. Os candidatos precisam de revisão "
            "visual antes de qualquer exclusão."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
