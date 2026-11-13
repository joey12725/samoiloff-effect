from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def test_app_renders_sample_print():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    captions = [i.caption for el in at.get("image") for i in el.proto.imgs]
    assert "The print under white light" in captions
    assert {"Under red light", "Under green light", "Under blue light"} <= set(captions)


def test_app_asks_for_uploads_without_samples():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception
    assert any("Add all three target images" in i.value for i in at.info)
