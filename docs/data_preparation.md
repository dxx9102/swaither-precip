# Data preparation

This document explains where to obtain the input data and how to prepare it
for use with SwAIther-Precip. The actual preprocessing pipeline is documented
and executable in [`notebooks/01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb). Please also see the paper (referenced in the README) for additional information. 

## Required inputs

The training and evaluation pipeline uses three data sources, all of which
must be obtained separately due to licensing restrictions.

### 1. AIFS forecasts (ECMWF)

**What it is:** The Artificial Intelligence Forecasting System (AIFS) is an
AI-based weather forecasting system developed by ECMWF. We use the
deterministic version (AIFS-Single 1.0) at its native 0.25° resolution as the
low-resolution input to our downscaling pipeline. The model can be used from Huggingface checkpoints (https://huggingface.co/ecmwf/aifs-single-1.0)

**Where to get it:** See [ecmwf.int/en/forecasts/datasets/aifs](https://www.ecmwf.int/en/forecasts/datasets/aifs) for general data access.
Access policies vary by user category (research, operational, commercial). Note that the AIFS forecasts dataset used in the SwAIther-Precip paper was built by the authors using the aifs-single-1.0 model and IFS initial conditions, and is not available for the public at the moment. We may release it publically at some point. 

**What we use:** Forecasts over the domain covering Switzerland, with 24
lead-time horizons at 6-hourly resolution (0–6 h, 6–12 h, ..., 138–144 h).
For each lead time, we use 11 channels:

- `total_precipitation`, `convective_precipitation` (in mm/6 h after preprocessing)
- `total_cloud_cover`
- `wind_10m_u`, `wind_10m_v`
- `specific_humidity_500hPa`, `specific_humidity_850hPa`
- `temperature_500hPa`, `temperature_850hPa`
- `geopotential_500hPa`, `geopotential_850hPa`

### 2. CombiPrecip observations (MeteoSwiss)

**What it is:** CombiPrecip is a radar-gauge combined precipitation analysis
produced operationally by MeteoSwiss at 1 km / 1 h resolution over
Switzerland. We use 6-hour accumulations (00 UTC and 12 UTC reference times)
as the high-resolution training target.

**Where to get it:** Contact [MeteoSwiss](https://www.meteoswiss.admin.ch/)
directly. CombiPrecip is licensed for non-commercial research use.

**What we use:** 6-hour cumulative precipitation fields for 2019–2023 over
the Swiss domain.

### 3. DHM25 digital elevation model (SwissTopo)

**What it is:** The DHM25 is a 25 m resolution digital elevation model of
Switzerland published by SwissTopo.

**Where to get it:** Available under an open license from
[swisstopo.admin.ch](https://www.swisstopo.admin.ch/).
We use the ASCII grid format (`dhm25_grid_raster.asc`).

**What we use:** A single static elevation field, reprojected onto each
training dataset's grid and appended as an `altitude` channel.

## Preprocessing pipeline

Once you have the raw inputs in `./data/`, the preprocessing pipeline is
documented and executable in
[`notebooks/01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb).
The notebook produces the following output files, which are referenced
throughout the training configurations in `conf/`:

| Output file | Used by |
|---|---|
| `AIFS_Single_pred_concat_2019_2023_all_0_24Horizons_NoNeg_andLowResPrecipWetEnough_gamma.nc` | Step 1 multi-lead training (main FiLM model, MultiFineTuned) |
| `AIFS_Single_pred_concat_2019_2023_SingleHori_6h_NoNeg_andLowResPrecipWetEnough_gamma.nc` | Step 1 single-lead 6 h baseline |
| `AIFS_Single_pred_concat_2019_2023_SingleHori_3day_NoNeg_andLowResPrecipWetEnough_gamma.nc` | Step 1 single-lead 3 day baseline + Strategy 2 ablation |
| `AIFS_Single_pred_concat_2019_2023_SingleHori_6day_NoNeg_andLowResPrecipWetEnough_gamma.nc` | Step 1 single-lead 6 day baseline |
| `CPCH_processed_precipitation_2019_2023_concat_NotFlat_6hSUM_00_12_wetEnough_gamma_wElevation.nc` | Step 2 super-resolution target (1 km CombiPrecip with elevation) |
| `stats/stats_aifs_single_*.json` | Per-config normalization statistics |

The pipeline consists of (see the notebook for full details):

1. **Preprocess AIFS forecasts** — convert precipitation channels to mm/h,
   enforce non-negativity, rename time coordinate.
2. **Preprocess CombiPrecip observations** — filter out near-zero-rain time
   steps (>99.5% zero pixels), cap per-image outliers using a fitted gamma
   distribution, harmonize naming and coordinates.
3. **Coarsen CombiPrecip onto the AIFS grid** — mass-conservative block-mean
   aggregation (linear interpolation is NOT used because it doesn't preserve
   total precipitation mass).
4. **Merge AIFS + coarse CombiPrecip** — pair along the common time axis, save
   multi-horizon and single-horizon variants.
5. **Add elevation** — reproject the SwissTopo DHM25 onto each dataset's grid
   and append it as a static `altitude` channel.

## Normalization statistics

Normalization statistics are stored as JSON files in `./data/stats/` and
referenced by every training and generation config via the `stats_path`
field. They are **dataset-specific**: each training configuration uses its
own stats file matched to the dataset variant it trains on (multi-horizon
vs. single-horizon, with vs. without elevation channel, Step 1 vs. Step 2).

We use **robust min/max scaling** based on the 0.5% and 99.5% percentiles
of each channel, computed on the training split only. The stats files are
JSON dicts of the form:

```json
{
  "input": {
    "total_precipitation_NoNeg": {"min": ..., "max": ...},
    ...
  },
  "output": {
    "precip6h_cumulative_capped": {"min": ..., "max": ...}
  }
}
```

> **If you are adapting SwAIther-Precip to your own data, you will need to
> compute new stats** matching the channels and time range of your training
> set. The published configs ship with stats files computed on our 2019–2023
> training subset; reusing them on a different dataset will produce
> degenerate outputs.

The stats files are produced by
[`notebooks/01_normalization_stats.ipynb`](../notebooks/01_normalization_stats.ipynb),
which computes them on the same 80/20 training split (seeded for
reproducibility) used by the training scripts. The schema is compatible
with PhysicsNeMo's CorrDiff example datasets — see the
[CorrDiff documentation](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff)
for general guidance on normalization conventions in this codebase.

## Directory layout

After preprocessing, your `./data/` directory should look like this:

```
data/
├── AIFS_Single_pred_concat_2019_2023_*.nc          # multi-horizon AIFS files
├── AIFS_Single_pred_concat_2019_2023_SingleHori_*.nc   # single-horizon AIFS files
├── CPCH_processed_precipitation_*_wElevation.nc    # CombiPrecip target with elevation
├── dhm25_grid_raster.asc                           # SwissTopo DEM (input only)
└── stats/
    ├── stats_aifs_single_all_0_24Horizons_wElev_biasCorrection.json
    ├── stats_aifs_single_STEP2_wElevation_allData.json
    ├── stats_aifs_single_6h_biasCorrection.json
    ├── stats_aifs_single_3day_biasCorrection.json
    └── stats_aifs_single_6day_biasCorrection.json
```

Convention: place all data in `./data/` at the repo root. If your data is
elsewhere on the filesystem, symlink it:

```bash
ln -s /path/to/your/data ./data
```

All training configs in `conf/` reference paths relative to `./data/`.
