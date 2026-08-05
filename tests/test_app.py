"""Smoke test: the packaged Streamlit app runs top to bottom (dashboard extra)."""

from importlib import resources

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_script_runs_clean():
    app_path = resources.files("dynamic_pricing_lab") / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Dynamic Pricing Lab"
    assert len(at.tabs) == 3
    assert len(at.sidebar) > 0
