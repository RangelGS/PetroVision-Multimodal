"""Cria o catálogo e diagnostica coincidências de identificador PPL/XPL."""

from __future__ import annotations

from petrovision.catalog import build_catalog, modality_pair_table
from petrovision.config import load_config, project_path


def main() -> int:
    config = load_config()
    raw_root = project_path(config["paths"]["raw_data"])
    output_path = project_path(config["paths"]["catalog"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog(
        raw_root=raw_root,
        modes=config["data"]["modes"],
        splits=config["data"]["splits"],
        extensions=config["data"]["extensions"],
    )

    if catalog.empty:
        print(f"Nenhuma imagem encontrada em: {raw_root}")
        print("Consulte a estrutura de pastas descrita no README.md.")
        return 1

    catalog.to_csv(output_path, index=False)
    comparison = modality_pair_table(catalog)
    overlap_path = output_path.with_name("modality_name_overlaps.csv")
    overlaps = comparison[comparison["has_pair"]].copy()
    overlaps.to_csv(overlap_path, index=False)

    print(f"Catálogo: {output_path}")
    print(f"Imagens: {len(catalog)}")
    print(f"Classes: {catalog['class_name'].nunique()}")
    print(
        "Coincidências de identificador PPL/XPL: "
        f"{len(overlaps)} (não interpretadas como pares petrográficos)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
