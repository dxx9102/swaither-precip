from __future__ import annotations

import pytest

from swaither.wind_helpers import configure_storage

configure_storage()

from swaither.datasets.wind import clear_cached_datasets


@pytest.fixture(autouse=True)
def _clear_wind_dataset_cache():
    clear_cached_datasets()
    yield
    clear_cached_datasets()
