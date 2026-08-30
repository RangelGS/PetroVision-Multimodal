"""Leitura e validação da configuração do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carrega o YAML e resolve os caminhos relativamente à raiz do projeto."""

    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("O arquivo de configuração deve conter um objeto YAML.")

    required_sections = {"paths", "data", "quality"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Seções ausentes em config.yaml: {sorted(missing)}")

    return config


def project_path(value: str | Path) -> Path:
    """Converte um caminho configurado em caminho absoluto do projeto."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

