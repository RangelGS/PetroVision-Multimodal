from dataclasses import dataclass
from io import BytesIO

import pytest
from remotezip import RemoteIOError

from petrovision.remote_data import (
    build_exact_pairs,
    download_image,
    find_stale_selection_images,
    image_selection_summary,
    local_relative_path,
    parse_archive_images,
    quarantine_existing_images,
    select_images,
    select_pairs,
    selection_summary,
    validate_disjoint_split_identities,
    validate_unique_hashes,
)


@dataclass
class FakeInfo:
    filename: str
    file_size: int = 100
    compress_size: int = 80


class FlakyRemoteZip:
    def __init__(self, payload: bytes, failures: int) -> None:
        self.payload = payload
        self.failures = failures
        self.calls = 0

    def open(self, _member_path: str, _mode: str) -> BytesIO:
        self.calls += 1
        if self.calls <= self.failures:
            raise RemoteIOError("504 Gateway Time-out")
        return BytesIO(self.payload)


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
    identities = [(item.mode, item.class_id, item.sample_id) for item in first]
    assert len(identities) == len(set(identities))


def test_unpaired_selection_reserves_train_ids_before_validation() -> None:
    infos = [
        member("PPL", "train", "class10", "sample.0"),
        member("PPL", "train", "class10", "sample.1"),
        member("PPL", "val", "class10", "sample.0"),
        member("PPL", "val", "class10", "sample.1"),
        member("PPL", "val", "class10", "sample.2"),
        member("PPL", "val", "class10", "sample.3"),
    ]
    images = parse_archive_images(
        infos,
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )

    selected = select_images(
        images,
        selected_classes={"class10": CLASSES["class10"]},
        images_per_split_per_mode={"train": 2, "val": 2},
        modes=["PPL"],
        seed=42,
    )

    train_ids = {item.sample_id for item in selected if item.split == "train"}
    val_ids = {item.sample_id for item in selected if item.split == "val"}
    assert train_ids == {"sample.0", "sample.1"}
    assert val_ids == {"sample.2", "sample.3"}
    assert train_ids.isdisjoint(val_ids)


def test_unpaired_selection_reports_shortage() -> None:
    with pytest.raises(ValueError, match="imagens suficientes"):
        select_images(
            [],
            selected_classes=CLASSES,
            images_per_split_per_mode={"train": 1},
            seed=42,
        )


def test_finds_images_removed_from_previous_selection() -> None:
    images = parse_archive_images(
        [
            member("PPL", "val", "class10", "old"),
            member("PPL", "val", "class10", "new"),
        ],
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )
    selected = [image for image in images if image.sample_id == "new"]

    stale = find_stale_selection_images(
        [image.member_path for image in images],
        selected,
        images,
    )

    assert [image.sample_id for image in stale] == ["old"]


def test_manifest_rejects_duplicate_content() -> None:
    digest = "a" * 64
    rows = [
        {"sha256": digest, "archive_member": "first.jpg"},
        {"sha256": digest, "archive_member": "second.jpg"},
    ]

    with pytest.raises(ValueError, match="Conteúdo duplicado"):
        validate_unique_hashes(rows)


def test_manifest_rejects_identity_reused_across_splits() -> None:
    rows = [
        {
            "mode": "PPL",
            "class_id": "class17",
            "sample_id": "Oolitic.20",
            "split": "train",
            "archive_member": "train/Oolitic.20.jpg",
        },
        {
            "mode": "PPL",
            "class_id": "class17",
            "sample_id": "Oolitic.20",
            "split": "val",
            "archive_member": "val/Oolitic.20.jpg",
        },
    ]

    with pytest.raises(ValueError, match="Identificador reutilizado"):
        validate_disjoint_split_identities(rows)


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


def test_download_retries_temporary_remote_error(tmp_path, monkeypatch) -> None:
    payload = b"x" * 100
    image = parse_archive_images(
        [member("PPL", "train", "class10", "Fracture.1")],
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )[0]
    remote = FlakyRemoteZip(payload, failures=2)
    delays: list[float] = []
    monkeypatch.setattr("petrovision.remote_data.time.sleep", delays.append)

    row = download_image(
        remote,
        image,
        project_root=tmp_path,
        max_attempts=3,
        retry_backoff_seconds=2.0,
    )

    destination = tmp_path / local_relative_path(image)
    assert remote.calls == 3
    assert delays == [2.0, 4.0]
    assert destination.read_bytes() == payload
    assert row["bytes"] == len(payload)
    assert not destination.with_suffix(".jpg.part").exists()


def test_download_removes_partial_after_retry_exhaustion(tmp_path, monkeypatch) -> None:
    image = parse_archive_images(
        [member("PPL", "train", "class10", "Fracture.2")],
        archive_prefix="carbonate_1223",
        selected_classes=CLASSES,
    )[0]
    remote = FlakyRemoteZip(b"x" * 100, failures=2)
    monkeypatch.setattr("petrovision.remote_data.time.sleep", lambda _delay: None)

    with pytest.raises(RemoteIOError, match="504 Gateway Time-out"):
        download_image(
            remote,
            image,
            project_root=tmp_path,
            max_attempts=2,
            retry_backoff_seconds=0.0,
        )

    destination = tmp_path / local_relative_path(image)
    assert remote.calls == 2
    assert not destination.exists()
    assert not destination.with_suffix(".jpg.part").exists()
