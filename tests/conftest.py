from __future__ import annotations

import pytest

from swaither.wind_helpers import configure_storage

configure_storage()


@pytest.fixture(autouse=True)
def _clear_wind_dataset_cache():
    yield
