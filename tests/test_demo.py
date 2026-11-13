from pathlib import Path

import numpy as np
from PIL import Image

import demo


def test_demo_writes_full_export(tmp_path):
    files = demo.run(tmp_path, canvas_in=(1.0, 0.75))
    assert (tmp_path / "print_white_light.png").exists()
    assert len(files) == 9
    assert Image.open(tmp_path / "print_white_light.png").size == (300, 225)


def test_committed_samples_match_generator():
    for img, name in zip(demo.sample_targets(), demo.SAMPLE_FILES, strict=True):
        committed = Image.open(Path(demo.SAMPLE_DIR) / name).convert("RGB")
        assert np.array_equal(np.asarray(committed), np.asarray(img)), name
