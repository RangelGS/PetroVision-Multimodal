"""Descoberta, catalogação e pareamento de imagens petrográficas."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CATALOG_COLUMNS = [
    "sample_id",
    "mode",
    "split",
    "class_name",
    "path",
    "extension",
    "file_size_bytes",
]


def _normalise_extensions(extensions: Iterable[str]) -> set[str]:
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }


def build_catalog(
    raw_root: str | Path,
    modes: Iterable[str],
    splits: Iterable[str],
    extensions: Iterable[str],
) -> pd.DataFrame:
    """Percorre `modo/divisão/classe/imagem` e retorna um catálogo ordenado."""

    raw_root = Path(raw_root)
    accepted = _normalise_extensions(extensions)
    records: list[dict[str, object]] = []

    for mode in modes:
        for split in splits:
            split_dir = raw_root / mode / split
            if not split_dir.exists():
                continue

            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.rglob("*")):
                    if not image_path.is_file() or image_path.suffix.lower() not in accepted:
                        continue

                    records.append(
                        {
                            "sample_id": image_path.stem,
                            "mode": str(mode),
                            "split": str(split),
                            "class_name": class_dir.name,
                            "path": str(image_path.resolve()),
                            "extension": image_path.suffix.lower(),
                            "file_size_bytes": image_path.stat().st_size,
                        }
                    )

    catalog = pd.DataFrame.from_records(records, columns=CATALOG_COLUMNS)
    if catalog.empty:
        return catalog

    duplicates = catalog.duplicated(
        subset=["sample_id", "mode", "split", "class_name"], keep=False
    )
    if duplicates.any():
        duplicated_rows = catalog.loc[duplicates, [
            "sample_id", "mode", "split", "class_name", "path"
        ]]
        raise ValueError(
            "Existem identificadores duplicados dentro da mesma modalidade, "
            f"classe e divisão:\n{duplicated_rows.to_string(index=False)}"
        )

    return catalog.sort_values(
        ["split", "class_name", "sample_id", "mode"]
    ).reset_index(drop=True)


def modality_pair_table(
    catalog: pd.DataFrame,
    first_mode: str = "PPL",
    second_mode: str = "XPL",
) -> pd.DataFrame:
    """Cria uma linha por amostra e indica se o par multimodal existe."""

    required = {"sample_id", "mode", "split", "class_name", "path"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no catálogo: {sorted(missing)}")

    filtered = catalog[catalog["mode"].isin([first_mode, second_mode])]
    paired = filtered.pivot(
        index=["sample_id", "split", "class_name"],
        columns="mode",
        values="path",
    ).reset_index()
    paired.columns.name = None

    for mode in (first_mode, second_mode):
        if mode not in paired:
            paired[mode] = pd.NA

    paired = paired.rename(
        columns={first_mode: f"path_{first_mode}", second_mode: f"path_{second_mode}"}
    )
    paired["has_pair"] = (
        paired[f"path_{first_mode}"].notna() & paired[f"path_{second_mode}"].notna()
    )
    return paired.sort_values(["split", "class_name", "sample_id"]).reset_index(drop=True)

