"""Mostra se o ambiente local está pronto para a primeira etapa."""

from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "cv2": "OpenCV",
    "yaml": "PyYAML",
    "sklearn": "Scikit-learn",
}


def main() -> int:
    print("PetroVision Multimodal - diagnóstico do ambiente")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.system()} {platform.release()}")

    failures = 0
    for module_name, display_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "versão não informada")
            print(f"[OK] {display_name}: {version}")
        except ImportError:
            failures += 1
            print(f"[FALTA] {display_name}")

    if failures:
        print("\nExecute: pip install -r requirements.txt")
        return 1

    print("\nAmbiente pronto para catálogo e controle de qualidade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

