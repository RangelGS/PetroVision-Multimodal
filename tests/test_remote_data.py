from dataclasses import dataclass

import pytest

from petrovision.remote_data import (
    build_exact_pairs,
    image_selection_summary,
    local_relative_path,
    parse_archive_images,
    quarantine_existing_images,
    select_images,
    select_pairs,
    selection_summary,
    validate_unique_hashes,
)


@dataclass
class FakeInfo:
    filename: str
    file_size: int = 100
    compress_size: int = 80


CLASSES = {
    "class10": "Cemented fracture",
    "class13": "Micritic limestone",
}


def member(mode: str, split: str, class_id: str, name: str) -> FakeInfo:
    return FakeInfo(f"carbonate_1223/{mode}-1223/{split}/{class_id}/{name}.jpg")


def test_parser_uses_class_directory_and_excludes_ars() -> None:
    infos = [
        member("PPL", "train", "class10", "LegacyFilename.1"),
        member("XPL", "train", "class10", "LegacyFilename.1"),
        member("PPL", "train", "class10", "LegacyFilename.2_ARS"),
        member("PPL", "train", "class1", "WrongSelectedClass.1"),
    ]
    images = parse_archive_images(
        infos,
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
        excluded_name_tokens=["_ARS"],
    )

    assert len(images) == 2
    assert {image.class_label for image in images} == {"Cemented fracture"}
    assert {image.mode for image in images} == {"PPL", "XPL"}


def test_parser_excludes_known_archive_member() -> None:
    excluded = "carbonate_1223/PPL-1223/train/class10/LegacyFilename.1.jpg"
    images = parse_archive_images(
        [member("PPL", "train", "class10", "LegacyFilename.1")],
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
        excluded_member_paths=[excluded],
    )

    assert images == []


def test_exact_pairing_discards_unpaired_image() -> None:
    infos = [
        member("PPL", "test", "class13", "Micritic.1"),
        member("XPL", "test", "class13", "Micritic.1"),
        member("PPL", "test", "class13", "Micritic.2"),
    ]
    images = parse_archive_images(
        infos,
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )
    pairs = build_exact_pairs(images)

    assert len(pairs) == 1
    assert pairs[0].sample_id == "Micritic.1"


def test_deterministic_stratified_selection() -> None:
    infos = []
    for class_id in CLASSES:
        for split in ("train", "val", "test"):
            for index in range(4):
                infos.extend(
                    [
                        member("PPL", split, class_id, f"sample.{index}"),
                        member("XPL", split, class_id, f"sample.{index}"),
                    ]
                )
    images = parse_archive_images(
        infos,
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )
    pairs = build_exact_pairs(images)
    requested = {"train": 2, "val": 1, "test": 1}
    first = select_pairs(
        pairs,
        selected_classes=CLASSES,
        pairs_per_split=requested,
        seed=42,
    )
    second = select_pairs(
        reversed(pairs),
        selected_classes=CLASSES,
        pairs_per_split=requested,
        seed=42,
    )
    counts, image_count, total_bytes = selection_summary(first)

    assert first == second
    assert len(first) == 8
    assert image_count == 16
    assert total_bytes == 1600
    assert counts[("class10", "train")] == 2
    class10_pair = next(pair for pair in first if pair.class_id == "class10")
    assert "class10_cemented_fracture" in str(
        local_relative_path(class10_pair.ppl)
    )


def test_selection_reports_shortage() -> None:
    with pytest.raises(ValueError, match="pares suficientes"):
        select_pairs(
            [],
            selected_classes=CLASSES,
            pairs_per_split={"train": 1},
            seed=42,
        )


def test_deterministic_unpaired_selection_by_mode() -> None:
    infos = []
    for class_id in CLASSES:
        for mode in ("PPL", "XPL"):
            for split in ("train", "val", "test"):
                for index in range(4):
                    infos.append(
                        member(mode, split, class_id, f"{mode.lower()}.{index}")
                    )
    images = parse_archive_images(
        infos,
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )
    requested = {"train": 2, "val": 1, "test": 1}
    first = select_images(
        images,
        selected_classes=CLASSES,
        images_per_split_per_mode=requested,
        seed=42,
    )
    second = select_images(
        reversed(images),
        selected_classes=CLASSES,
        images_per_split_per_mode=requested,
        seed=42,
    )
    counts, image_count, total_bytes = image_selection_summary(first)

    assert first == second
    assert image_count == 16
    assert total_bytes == 1600
    assert counts[("PPL", "class10", "train")] == 2
    assert counts[("XPL", "class13", "test")] == 1


def test_unpaired_selection_reports_shortage() -> None:
    with pytest.raises(ValueError, match="imagens suficientes"):
        select_images(
            [],
            selected_classes=CLASSES,
            images_per_split_per_mode={"train": 1},
            seed=42,
        )


def test_manifest_rejects_duplicate_content() -> None:
    digest = "a" * 64
    rows = [
        {"sha256": digest, "archive_member": "first.jpg"},
        {"sha256": digest, "archive_member": "second.jpg"},
    ]

    with pytest.raises(ValueError, match="Conteúdo duplicado"):
        validate_unique_hashes(rows)


def test_quarantine_moves_existing_raw_file(tmp_path) -> None:
    image = parse_archive_images(
        [member("PPL", "train", "class10", "Fracture.999")],
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )[0]
    source = tmp_path / local_relative_path(image)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"duplicate-image")
    quarantine_root = tmp_path / "data" / "interim" / "quarantine"

    moves = quarantine_existing_images(
        [image],
        project_root=tmp_path,
        quarantine_root=quarantine_root,
    )

    assert len(moves) == 1
    assert not source.exists()
    assert moves[0][1].read_bytes() == b"duplicate-image"
