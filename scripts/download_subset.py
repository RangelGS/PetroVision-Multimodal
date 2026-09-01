"""Inspeciona e baixa o subconjunto PPL/XPL não pareado do DeepCarbonate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

from petrovision.config import PROJECT_ROOT, load_config, project_path
from petrovision.remote_data import (
    build_exact_pairs,
    download_image,
    find_stale_selection_images,
    image_selection_summary,
    parse_archive_images,
    quarantine_existing_images,
    select_images,
    validate_disjoint_split_identities,
    validate_unique_hashes,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona imagens PPL/XPL estratificadas sem baixar o ZIP completo "
            "do DeepCarbonate."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuração relativa à raiz do projeto (padrão: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra disponibilidade, seleção e tamanho sem baixar imagens.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma o download sem pergunta interativa.",
    )
    return parser.parse_args()


def format_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("Unreachable")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    remote = config.get("remote_dataset")
    if not isinstance(remote, dict):
        raise ValueError("Seção remote_dataset ausente ou inválida no config.yaml.")

    selected_config = remote["selected_classes"]
    selected_classes = {
        class_id: values["label"] for class_id, values in selected_config.items()
    }
    requested = {
        split: int(count)
        for split, count in remote["images_per_split_per_mode"].items()
    }

    print("Conectando ao DeepCarbonate e lendo o índice remoto...")
    archive = RemoteZip(
        remote["archive_url"],
        support_suffix_range=False,
        initial_buffer_size=int(remote["initial_buffer_mb"]) * 1024 * 1024,
        timeout=int(remote["timeout_seconds"]),
    )
    try:
        archive_infos = archive.infolist()
        all_images = parse_archive_images(
            archive_infos,
            archive_prefix=remote["archive_prefix"],
            selected_classes=selected_classes,
            excluded_name_tokens=remote.get("exclude_name_contains", []),
        )
        excluded_members = set(remote.get("exclude_archive_members", []))
        excluded_images = [
            image for image in all_images if image.member_path in excluded_members
        ]
        found_exclusions = {image.member_path for image in excluded_images}
        missing_exclusions = excluded_members.difference(found_exclusions)
        if missing_exclusions:
            raise ValueError(
                "Entradas configuradas para exclusão não foram encontradas:\n- "
                + "\n- ".join(sorted(missing_exclusions))
            )
        images = [
            image for image in all_images if image.member_path not in excluded_members
        ]
        exact_name_pairs = build_exact_pairs(images)
        selected = select_images(
            images,
            selected_classes=selected_classes,
            images_per_split_per_mode=requested,
            modes=config["data"]["modes"],
            seed=int(config["project"]["random_seed"]),
        )
        counts, image_count, total_bytes = image_selection_summary(selected)
        manifest_path = project_path(remote["manifest"])
        previous_members: set[str] = set()
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8", newline="") as stream:
                previous_members = {
                    str(row.get("archive_member", ""))
                    for row in csv.DictReader(stream)
                    if row.get("archive_member")
                }
        stale_images = find_stale_selection_images(
            previous_members,
            selected,
            all_images,
        )

        print("\nSeleção determinística confirmada:")
        print("classe  rótulo oficial                 modo  treino  validação  teste  total")
        for class_id, label in selected_classes.items():
            for mode in config["data"]["modes"]:
                train = counts.get((mode, class_id, "train"), 0)
                val = counts.get((mode, class_id, "val"), 0)
                test = counts.get((mode, class_id, "test"), 0)
                print(
                    f"{class_id:<8}{label:<31}{mode:<6}{train:>6}"
                    f"{val:>11}{test:>7}{train + val + test:>7}"
                )
        print(f"\nTotal: {image_count} imagens")
        print(f"Tamanho descompactado estimado: {format_size(total_bytes)}")
        print("Imagens com sufixo _ARS: excluídas")
        print(
            "Entradas com conteúdo duplicado conhecido: "
            f"{len(excluded_images)} excluídas da seleção"
        )
        print(
            "Correspondências exatas de nome PPL/XPL: "
            f"{len(exact_name_pairs)} (somente diagnóstico; não usadas como pares)"
        )
        print(
            "Estratégia: modalidades não pareadas, estratificadas de forma "
            "independente."
        )
        print(
            "Imagens de uma seleção anterior a mover para quarentena: "
            f"{len(stale_images)}"
        )

        if args.dry_run:
            print("\nSimulação concluída. Nenhuma imagem foi baixada.")
            return 0

        if not args.yes:
            answer = input("\nBaixar este subconjunto agora? [s/N] ").strip().casefold()
            if answer not in {"s", "sim", "y", "yes"}:
                print("Download cancelado. Nenhum arquivo foi alterado.")
                return 0

        rows: list[dict[str, object]] = []
        for image in tqdm(selected, desc="Baixando", unit="imagem"):
            rows.append(
                download_image(archive, image, project_root=Path(PROJECT_ROOT))
            )

        validate_unique_hashes(rows)
        validate_disjoint_split_identities(rows)
        quarantine_root = project_path(remote["quarantine_root"])
        moves = quarantine_existing_images(
            excluded_images,
            project_root=Path(PROJECT_ROOT),
            quarantine_root=quarantine_root,
        )
        stale_moves = quarantine_existing_images(
            stale_images,
            project_root=Path(PROJECT_ROOT),
            quarantine_root=project_path(remote["superseded_quarantine_root"]),
        )
        write_manifest(
            rows,
            path=manifest_path,
            dataset_doi=str(remote["doi"]),
        )
        print(f"\nDownload concluído. Manifesto: {manifest_path}")
        print(f"Arquivos movidos para quarentena: {len(moves)}")
        print(f"Arquivos substituídos movidos para quarentena: {len(stale_moves)}")
        print("Os dados brutos continuam ignorados pelo Git.")
        return 0
    finally:
        archive.close()


if __name__ == "__main__":
    raise SystemExit(main())
