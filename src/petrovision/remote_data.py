"""Seleção multimodal reprodutível e download parcial do DeepCarbonate."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from remotezip import RemoteIOError


@dataclass(frozen=True)
class ArchiveImage:
    """Imagem reconhecida no diretório interno do arquivo remoto."""

    member_path: str
    mode: str
    split: str
    class_id: str
    class_label: str
    filename: str
    sample_id: str
    file_size: int
    compress_size: int


@dataclass(frozen=True)
class PairedSample:
    """Par da mesma amostra observado em PPL e XPL."""

    split: str
    class_id: str
    class_label: str
    sample_id: str
    ppl: ArchiveImage
    xpl: ArchiveImage


MANIFEST_COLUMNS = [
    "dataset_doi",
    "class_id",
    "class_label",
    "split",
    "sample_id",
    "mode",
    "archive_member",
    "local_relative_path",
    "bytes",
    "sha256",
]


def _archive_pattern(prefix: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(prefix)}/(?P<mode>PPL|XPL)-1223/"
        rf"(?P<split>train|val|test)/(?P<class_id>class\d+)/"
        rf"(?P<filename>[^/]+\.(?:jpg|jpeg|png|tif|tiff))$",
        flags=re.IGNORECASE,
    )


def parse_archive_images(
    infos: Iterable[object],
    *,
    archive_prefix: str,
    selected_classes: Mapping[str, str],
    excluded_name_tokens: Sequence[str] = ("_ARS",),
    excluded_member_paths: Sequence[str] = (),
) -> list[ArchiveImage]:
    """Converte entradas ZipInfo em imagens válidas das classes escolhidas.

    A classe vem exclusivamente do diretório ``classN`` e do ``classmap``.
    O nome histórico do arquivo não é usado como rótulo.
    """

    pattern = _archive_pattern(archive_prefix.strip("/"))
    excluded = tuple(token.casefold() for token in excluded_name_tokens)
    excluded_members = {path.casefold() for path in excluded_member_paths}
    normalised_classes = {
        class_id.casefold(): (class_id, label)
        for class_id, label in selected_classes.items()
    }
    images: list[ArchiveImage] = []

    for info in infos:
        member_path = str(getattr(info, "filename", ""))
        if member_path.casefold() in excluded_members:
            continue
        match = pattern.fullmatch(member_path)
        if not match:
            continue

        filename = match.group("filename")
        if any(token in filename.casefold() for token in excluded):
            continue

        class_key = match.group("class_id").casefold()
        if class_key not in normalised_classes:
            continue
        class_id, class_label = normalised_classes[class_key]

        images.append(
            ArchiveImage(
                member_path=member_path,
                mode=match.group("mode").upper(),
                split=match.group("split").lower(),
                class_id=class_id,
                class_label=class_label,
                filename=filename,
                sample_id=Path(filename).stem,
                file_size=int(getattr(info, "file_size", 0)),
                compress_size=int(getattr(info, "compress_size", 0)),
            )
        )

    return sorted(
        images,
        key=lambda item: (
            item.split,
            item.class_id,
            item.sample_id.casefold(),
            item.mode,
        ),
    )


def build_exact_pairs(images: Iterable[ArchiveImage]) -> list[PairedSample]:
    """Mantém somente amostras que possuem correspondência exata PPL/XPL."""

    grouped: dict[tuple[str, str, str], dict[str, ArchiveImage]] = {}
    for image in images:
        key = (image.split, image.class_id, image.sample_id.casefold())
        modes = grouped.setdefault(key, {})
        if image.mode in modes:
            raise ValueError(
                "Entrada duplicada no arquivo remoto para "
                f"{image.mode}/{image.split}/{image.class_id}/{image.sample_id}."
            )
        modes[image.mode] = image

    pairs: list[PairedSample] = []
    for (split, class_id, _), modes in grouped.items():
        if set(modes) != {"PPL", "XPL"}:
            continue
        ppl = modes["PPL"]
        xpl = modes["XPL"]
        pairs.append(
            PairedSample(
                split=split,
                class_id=class_id,
                class_label=ppl.class_label,
                sample_id=ppl.sample_id,
                ppl=ppl,
                xpl=xpl,
            )
        )

    return sorted(
        pairs,
        key=lambda item: (item.split, item.class_id, item.sample_id.casefold()),
    )


def select_pairs(
    pairs: Iterable[PairedSample],
    *,
    selected_classes: Mapping[str, str],
    pairs_per_split: Mapping[str, int],
    seed: int,
) -> list[PairedSample]:
    """Escolhe quantidades fixas por classe/divisão usando ranking SHA-256."""

    grouped: dict[tuple[str, str], list[PairedSample]] = {}
    for pair in pairs:
        grouped.setdefault((pair.split, pair.class_id), []).append(pair)

    selected: list[PairedSample] = []
    shortages: list[str] = []
    for class_id in selected_classes:
        for split, requested in pairs_per_split.items():
            candidates = grouped.get((split, class_id), [])
            if len(candidates) < requested:
                shortages.append(
                    f"{class_id}/{split}: {len(candidates)} disponíveis, "
                    f"{requested} solicitados"
                )
                continue

            def rank(pair: PairedSample) -> str:
                payload = f"{seed}|{class_id}|{split}|{pair.sample_id}".encode()
                return hashlib.sha256(payload).hexdigest()

            selected.extend(sorted(candidates, key=rank)[:requested])

    if shortages:
        raise ValueError(
            "O arquivo remoto não contém pares suficientes:\n- "
            + "\n- ".join(shortages)
        )

    return sorted(
        selected,
        key=lambda item: (item.split, item.class_id, item.sample_id.casefold()),
    )


def selection_summary(
    selected: Iterable[PairedSample],
) -> tuple[dict[tuple[str, str], int], int, int]:
    """Retorna pares por classe/divisão, imagens e bytes descompactados."""

    counts: dict[tuple[str, str], int] = {}
    images = 0
    total_bytes = 0
    for pair in selected:
        counts[(pair.class_id, pair.split)] = (
            counts.get((pair.class_id, pair.split), 0) + 1
        )
        images += 2
        total_bytes += pair.ppl.file_size + pair.xpl.file_size
    return counts, images, total_bytes


def select_images(
    images: Iterable[ArchiveImage],
    *,
    selected_classes: Mapping[str, str],
    images_per_split_per_mode: Mapping[str, int],
    modes: Sequence[str] = ("PPL", "XPL"),
    seed: int,
) -> list[ArchiveImage]:
    """Seleciona imagens sem presumir correspondência individual entre modos.

    O DeepCarbonate publica PPL e XPL em subconjuntos separados, mas não fornece
    uma chave de pareamento no ZIP. Por isso, a estratificação é independente
    para cada combinação de modalidade, classe e divisão oficial.
    """

    grouped: dict[tuple[str, str, str], list[ArchiveImage]] = {}
    for image in images:
        grouped.setdefault((image.mode, image.split, image.class_id), []).append(image)

    split_priority = ("train", "val", "test")
    ordered_splits = [
        split for split in split_priority if split in images_per_split_per_mode
    ]
    ordered_splits.extend(
        split
        for split in images_per_split_per_mode
        if split not in split_priority
    )

    selected: list[ArchiveImage] = []
    shortages: list[str] = []
    for class_id in selected_classes:
        for mode in modes:
            normalised_mode = mode.upper()
            reserved_ids: set[str] = set()
            for split in ordered_splits:
                requested = images_per_split_per_mode[split]
                candidates = grouped.get((normalised_mode, split, class_id), [])

                def rank(image: ArchiveImage) -> str:
                    payload = (
                        f"{seed}|{normalised_mode}|{class_id}|{split}|"
                        f"{image.sample_id}"
                    ).encode()
                    return hashlib.sha256(payload).hexdigest()

                available: list[ArchiveImage] = []
                seen_in_split: set[str] = set()
                for image in sorted(candidates, key=rank):
                    identity = image.sample_id.casefold()
                    if identity in reserved_ids or identity in seen_in_split:
                        continue
                    seen_in_split.add(identity)
                    available.append(image)

                if len(available) < requested:
                    shortages.append(
                        f"{normalised_mode}/{class_id}/{split}: "
                        f"{len(available)} identificadores únicos disponíveis após "
                        f"reservar divisões anteriores, {requested} solicitadas"
                    )
                    continue

                chosen = available[:requested]
                selected.extend(chosen)
                reserved_ids.update(
                    image.sample_id.casefold() for image in chosen
                )

    if shortages:
        raise ValueError(
            "O arquivo remoto não contém imagens suficientes:\n- "
            + "\n- ".join(shortages)
        )

    return sorted(
        selected,
        key=lambda item: (
            item.mode,
            item.split,
            item.class_id,
            item.sample_id.casefold(),
        ),
    )


def image_selection_summary(
    selected: Iterable[ArchiveImage],
) -> tuple[dict[tuple[str, str, str], int], int, int]:
    """Retorna imagens por modalidade/classe/divisão, total e bytes."""

    counts: dict[tuple[str, str, str], int] = {}
    image_count = 0
    total_bytes = 0
    for image in selected:
        key = (image.mode, image.class_id, image.split)
        counts[key] = counts.get(key, 0) + 1
        image_count += 1
        total_bytes += image.file_size
    return counts, image_count, total_bytes


def find_stale_selection_images(
    previous_archive_members: Iterable[str],
    selected: Iterable[ArchiveImage],
    available_images: Iterable[ArchiveImage],
) -> list[ArchiveImage]:
    """Localiza imagens de um manifesto anterior que saíram da seleção atual."""

    selected_members = {image.member_path for image in selected}
    stale_members = set(previous_archive_members).difference(selected_members)
    available_by_member = {image.member_path: image for image in available_images}
    unknown = stale_members.difference(available_by_member)
    if unknown:
        raise ValueError(
            "O manifesto anterior contém entradas ausentes no índice remoto:\n- "
            + "\n- ".join(sorted(unknown))
        )
    return sorted(
        (available_by_member[member] for member in stale_members),
        key=lambda image: image.member_path,
    )


def class_slug(class_id: str, class_label: str) -> str:
    """Gera diretório portátil que preserva o identificador oficial da classe."""

    label = re.sub(r"[^a-z0-9]+", "_", class_label.casefold()).strip("_")
    return f"{class_id}_{label}"


def local_relative_path(image: ArchiveImage) -> Path:
    """Define o destino local seguro de uma imagem selecionada."""

    safe_name = Path(image.filename).name
    if safe_name != image.filename or safe_name in {"", ".", ".."}:
        raise ValueError(f"Nome de arquivo inseguro: {image.filename!r}")
    return (
        Path("data")
        / "raw"
        / image.mode
        / image.split
        / class_slug(image.class_id, image.class_label)
        / safe_name
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unique_hashes(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Impede que conteúdos idênticos entrem no subconjunto final."""

    materialized = list(rows)
    by_hash: dict[str, list[Mapping[str, object]]] = {}
    for row in materialized:
        digest = str(row.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"SHA-256 inválido no manifesto: {digest!r}")
        by_hash.setdefault(digest, []).append(row)

    duplicates = {digest: group for digest, group in by_hash.items() if len(group) > 1}
    if duplicates:
        details = []
        for digest, group in duplicates.items():
            members = ", ".join(str(row.get("archive_member", "?")) for row in group)
            details.append(f"{digest}: {members}")
        raise ValueError(
            "Conteúdo duplicado detectado no subconjunto final:\n- "
            + "\n- ".join(details)
        )
    return materialized


def validate_disjoint_split_identities(
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Impede que um identificador seja reutilizado entre divisões.

    A identidade é definida dentro da mesma modalidade e classe. O mesmo nome
    em PPL e XPL continua permitido porque o ZIP não fornece pareamento seguro
    entre essas modalidades.
    """

    materialized = list(rows)
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in materialized:
        mode = str(row.get("mode", "")).strip().upper()
        class_id = str(row.get("class_id", "")).strip().casefold()
        sample_id = str(row.get("sample_id", "")).strip().casefold()
        split = str(row.get("split", "")).strip().casefold()
        if not all((mode, class_id, sample_id, split)):
            raise ValueError(
                "Não foi possível auditar identidades: mode, class_id, "
                "sample_id e split são obrigatórios."
            )
        grouped.setdefault((mode, class_id, sample_id), []).append(row)

    conflicts: list[str] = []
    for (mode, class_id, sample_id), group in grouped.items():
        splits = {str(row["split"]).strip().casefold() for row in group}
        if len(splits) <= 1:
            continue
        members = ", ".join(
            f"{row['split']}:{row.get('archive_member', '?')}" for row in group
        )
        conflicts.append(f"{mode}/{class_id}/{sample_id}: {members}")

    if conflicts:
        raise ValueError(
            "Identificador reutilizado entre treino, validação ou teste:\n- "
            + "\n- ".join(conflicts)
        )
    return materialized


def quarantine_existing_images(
    images: Iterable[ArchiveImage],
    *,
    project_root: Path,
    quarantine_root: Path,
) -> list[tuple[Path, Path]]:
    """Move arquivos excluídos de ``data/raw`` para quarentena recuperável."""

    moves: list[tuple[Path, Path]] = []
    for image in images:
        relative_source = local_relative_path(image)
        source = project_root / relative_source
        if not source.exists():
            continue

        raw_subpath = Path(*relative_source.parts[2:])
        destination = quarantine_root / raw_subpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256(source) != _sha256(destination):
                raise FileExistsError(
                    f"Conflito na quarentena: {destination} já existe com outro conteúdo."
                )
            source.unlink()
        else:
            os.replace(source, destination)
        moves.append((source, destination))
    return moves


def download_image(
    remote_zip: object,
    image: ArchiveImage,
    *,
    project_root: Path,
    chunk_size: int = 1024 * 1024,
    max_attempts: int = 1,
    retry_backoff_seconds: float = 0.0,
) -> dict[str, object]:
    """Baixa uma entrada com escrita atômica e retentativas de rede."""

    if max_attempts < 1:
        raise ValueError("max_attempts deve ser pelo menos 1.")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds não pode ser negativo.")

    relative_path = local_relative_path(image)
    destination = project_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists() or destination.stat().st_size != image.file_size:
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, max_attempts + 1):
            try:
                with remote_zip.open(image.member_path, "r") as source, partial.open(
                    "wb"
                ) as out:
                    for chunk in iter(lambda: source.read(chunk_size), b""):
                        out.write(chunk)
                if partial.stat().st_size != image.file_size:
                    raise IOError(
                        f"Tamanho incorreto em {image.filename}: "
                        f"{partial.stat().st_size} != {image.file_size} bytes."
                    )
                os.replace(partial, destination)
                break
            except RemoteIOError as exc:
                if attempt >= max_attempts:
                    raise
                delay = min(retry_backoff_seconds * (2 ** (attempt - 1)), 30.0)
                print(
                    "\nFalha temporária ao baixar "
                    f"{image.filename} (tentativa {attempt}/{max_attempts}). "
                    f"Nova tentativa em {delay:.1f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
            finally:
                if partial.exists():
                    partial.unlink()

    return {
        "class_id": image.class_id,
        "class_label": image.class_label,
        "split": image.split,
        "sample_id": image.sample_id,
        "mode": image.mode,
        "archive_member": image.member_path,
        "local_relative_path": relative_path.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def write_manifest(
    rows: Iterable[Mapping[str, object]],
    *,
    path: Path,
    dataset_doi: str,
) -> None:
    """Grava o manifesto auditável em CSV."""

    validated_rows = validate_unique_hashes(rows)
    validate_disjoint_split_identities(validated_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in validated_rows:
                writer.writerow({"dataset_doi": dataset_doi, **row})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
