"""Extrai embeddings globais do DINOv2 para as imagens aprovadas."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from petrovision import __version__
from petrovision.config import PROJECT_ROOT, load_config, project_path
from petrovision.embeddings import (
    l2_normalize_embeddings,
    save_embedding_bundle,
    save_run_metadata,
    select_model_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    """Obtém o commit da execução sem impedir uso fora de um clone Git."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    args = parse_args()
    try:
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        print("Dependências de modelo ausentes.")
        print("No Colab, execute: pip install -r requirements-colab.txt")
        print(f"Detalhe: {exc}")
        return 1

    config = load_config(args.config)
    model_config = config["models"]["dinov2"]
    batch_size = args.batch_size or int(model_config["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size deve ser positivo.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    quality_path = project_path(config["paths"]["quality_report"])
    if not quality_path.exists():
        print("Relatório de qualidade não encontrado.")
        print("Execute primeiro: python scripts/run_quality_control.py")
        return 1

    config_path = project_path(args.config)
    manifest_path = project_path(config["remote_dataset"]["manifest"])
    if not manifest_path.exists():
        print("Manifesto do subconjunto não encontrado.")
        print("Execute primeiro: python scripts/download_subset.py --yes")
        return 1

    index = select_model_inputs(pd.read_csv(quality_path))
    missing_paths = [path for path in index["path"].map(Path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} imagens do índice não foram encontradas; "
            f"primeiro exemplo: {missing_paths[0]}"
        )

    model_id = str(model_config["model_id"])
    revision = str(model_config["revision"])
    print(f"Dispositivo: {device}")
    print(f"Modelo: {model_id}")
    print(f"Revisão fixa: {revision}")
    print(f"Imagens aprovadas: {len(index)}")

    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(index), batch_size), desc="DINOv2"):
        rows = index.iloc[start : start + batch_size]
        images: list[Image.Image] = []
        for path in rows["path"]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        chunks.append(cls_embeddings.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(chunks, axis=0)
    if bool(model_config.get("l2_normalize", True)):
        embeddings = l2_normalize_embeddings(embeddings)

    embeddings_path = project_path(config["paths"]["dinov2_embeddings"])
    index_path = project_path(config["paths"]["dinov2_index"])
    run_path = project_path(config["paths"]["dinov2_run"])
    save_embedding_bundle(embeddings, index, embeddings_path, index_path)
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "petrovision_version": __version__,
        "git_commit": git_commit(Path(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "subset_manifest_sha256": sha256_file(manifest_path),
        "model_id": model_id,
        "model_revision": revision,
        "embedding_source": str(model_config["embedding_source"]),
        "l2_normalized": bool(model_config.get("l2_normalize", True)),
        "samples": int(len(index)),
        "embedding_dimensions": int(embeddings.shape[1]),
        "batch_size": int(batch_size),
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "quality_report_sha256": sha256_file(quality_path),
        "embeddings_npz_sha256": sha256_file(embeddings_path),
    }
    save_run_metadata(run_metadata, run_path)

    print(f"Embeddings: {embeddings_path}")
    print(f"Índice: {index_path}")
    print(f"Metadados da execução: {run_path}")
    print(f"Formato final: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
