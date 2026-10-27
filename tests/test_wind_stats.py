from __future__ import annotations

import numpy as np

from scripts.compute_wind_stats import _dynamic_stats, _stats_from_store


class _FakeTimeIndex:
    lat = np.arange(2, dtype=np.float32)
    lon = np.arange(2, dtype=np.float32)

    @staticmethod
    def read_variables_many(variables, positions, lat, lon, engine):
        del variables, lat, lon, engine
        # (time, channel, lat, lon): the extreme value lives in the final
        # time/channel and must survive reservoir aggregation.
        values = np.zeros((len(positions), 2, 2, 2), dtype=np.float32)
        values[-1, 0] = 10.0
        values[-1, 1] = -20.0
        return values


def test_dynamic_stats_accumulates_every_time_and_channel():
    store = {}
    _dynamic_stats(
        _FakeTimeIndex(),
        ["u10", "v10"],
        np.arange(3, dtype=np.int64),
        store,
        limit=100,
        engine="netcdf4",
    )

    assert set(store) == {"u10", "v10"}
    assert float(store["u10"].max()) == 10.0
    assert float(store["v10"].min()) == -20.0
    result = _stats_from_store(store, 0.0, 1.0)
    assert result["u10"] == {"min": 0.0, "max": 10.0}
    assert result["v10"] == {"min": -20.0, "max": 0.0}
