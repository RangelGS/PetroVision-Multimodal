from pathlib import Path

import cv2
import numpy as np

import pandas as pd

from petrovision.quality import (
    apply_manual_reviews,
    apply_quality_rules,
    calculate_quality_metrics,
)


def test_uniform_image_is_flagged_for_low_contrast_and_blur(tmp_path: Path) -> None:
    path = tmp_path / "uniform.png"
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    metrics = calculate_quality_metrics(path)
    result = apply_quality_rules(
        metrics,
        {
            "min_brightness": 35.0,
            "max_brightness": 220.0,
            "min_contrast": 18.0,
            "min_sharpness": 60.0,
        },
    )

    assert metrics["width"] == 32
    assert metrics["height"] == 32
    assert result["automatic_quality_ok"] is False
    assert result["review_required"] is True
    assert result["automatic_status"] == "review_required"
    assert "baixo_contraste" in str(result["quality_flags"])
    assert "possivel_desfoque" in str(result["quality_flags"])


def test_manual_review_can_retain_a_flagged_image() -> None:
    report = pd.DataFrame(
        [
            {
                "mode": "XPL",
                "split": "val",
                "class_name": "class22_pore",
                "sample_id": "Vug.1753",
                "path": "image.jpg",
                "automatic_quality_ok": False,
                "read_error": "",
            }
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "mode": "XPL",
                "split": "val",
                "class_name": "class22_pore",
                "sample_id": "Vug.1753",
                "manual_decision": "keep",
                "review_notes": "Escura, mas válida após inspeção visual.",
            }
        ]
    )

    reviewed = apply_manual_reviews(report, reviews)

    assert reviewed.loc[0, "final_status"] == "manual_keep"
    assert bool(reviewed.loc[0, "use_for_model"]) is True
