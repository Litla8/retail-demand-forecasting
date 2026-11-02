# Retail Sales & Demand Forecasting

Forecast daily unit sales for every store and item for the next 90 days, so a retail chain can plan
inventory, staffing and promotions. Dataset: Kaggle *Store Item Demand Forecasting Challenge*
(10 stores x 50 items = 500 series, 2013-2017).

## Status

- [x] Step 1: Business understanding (project brief, metrics, config)
- [x] Step 2: Data preparation (validation, complete store-item-day grid, calendar features, time-based windows)
- [ ] Step 3: Time series EDA
- [ ] Step 4: Baseline models
- [ ] Step 5: Statistical models
- [ ] Step 6: Machine learning (LightGBM)
- [ ] Step 7: Tuning and ensembling
- [ ] Step 8: Business application
- [ ] Step 9: Dashboard and deployment

## Key decisions (Step 1)

- **Granularity:** store x item x day (500 series).
- **Horizon:** 90 days, matching the Kaggle test period (2018-01-01 to 2018-03-31); error is also
  reported for days 1-28, 29-60 and 61-90.
- **Primary metric:** WAPE. Also sMAPE (the Kaggle metric), MAE, RMSE and bias.
- **Validation:** time-based only. 90-day windows ending 2017-03-31 (same season as the test period),
  2017-06-30, 2017-09-30 and 2017-12-31.
- **Benchmark to beat:** the same weekday one year earlier (364-day seasonal naive).
- **Target:** 10-20% lower WAPE than the benchmark, stable across windows.

## Setup

Download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition into `data/raw/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python -m forecast.prepare    # validate, clean and write data/processed/sales_prepared.parquet
```

`python -m forecast.prepare` also writes `reports/data_quality.json` (row counts, date ranges, zero/negative/missing
days, memory before and after downcasting, validation windows). Flags: `--raw-dir`, `--processed-dir`,
`--reports-dir`, `--log-level DEBUG`.

## Structure

```
forecast/     config.py (all settings), metrics.py (WAPE, sMAPE, MAE, RMSE, bias, error by group),
              data.py (load, validate, clean, complete grid), features.py (calendar features),
              splits.py (time-based validation windows), prepare.py (Step 2 pipeline)
tests/        pytest suite
data/raw/     Kaggle files (git-ignored)
data/processed/  prepared parquet (git-ignored, recreated by forecast.prepare)
```
