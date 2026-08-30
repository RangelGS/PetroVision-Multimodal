from pathlib import Path

import cv2
import numpy as np

from petrovision.quality import apply_quality_rules, calculate_quality_metrics


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
    assert result["quality_ok"] is False
    assert "baixo_contraste" in str(result["quality_flags"])
    assert "possivel_desfoque" in str(result["quality_flags"])

