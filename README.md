# Retail Sales & Demand Forecasting

Forecast daily unit sales for every store and item for the next 90 days, so a retail chain can plan
inventory, staffing and promotions. Dataset: Kaggle *Store Item Demand Forecasting Challenge*
(10 stores x 50 items = 500 series, 2013-2017).

## Status

- [x] Step 1: Business understanding (project brief, metrics, config)
- [x] Step 2: Data preparation (validation, complete store-item-day grid, calendar features, time-based windows)
- [x] Step 3: Time series EDA (trend, seasonality, holidays, intermittency, ACF/PACF, STL, stationarity)
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

## Findings from the time series EDA (Step 3)

![Seasonality](docs/images/eda_seasonality.png)

- **Trend:** yearly sales grew 15.0% (2014), 4.4% (2015), 8.6% (2016) and 3.6% (2017). The 2017 growth is almost
  the same for every series (median 3.7%, 5th-95th percentile 1.9-5.4%), so one shared trend fits all of them.
- **Weekly pattern:** Monday is the low (0.79x the average) and Sunday the high (1.19x), a 50% swing, identical
  every year.
- **Yearly pattern is a staircase:** the level is flat inside a month (0.2% drift) and jumps by about 11.6% on
  average at every month boundary (January 0.68x, July 1.28x, December back down to 0.75x). Smooth yearly curves
  (Fourier terms, Prophet's default) will blur these steps, so month needs to be an explicit feature.
- **One seasonal shape for all series:** every series' weekday and month profile correlates at least 0.988 with the
  average profile, which supports a single global model.
- **No holiday effects:** all 18 tested holidays are within about 1% of a normal day (ordinary noise is 0.7%), so no
  holiday features are needed. A first version of this test showed a fake +15% effect on Thanksgiving: the
  comparison days fell in December, where sales drop 30% because of the month step. Comparing each day with the
  same weekday in the same month fixed it.
- **No intermittent demand:** all 500 series are classed smooth; there is one zero-sales day in 913,000 rows.
- **Autocorrelation:** lag 7 is 0.96 and lag 364 is 0.72 for total sales, but lag 365 is only 0.53, so the yearly
  cycle lines up on the same weekday (364 days), which supports the 364-day benchmark.

![ACF and PACF](docs/images/eda_acf_pacf.png)

Modelling implications: use month and day-of-week as explicit features, train one global model across all series,
model sales as level x seasonality x trend (log scale), add no holiday features, and skip intermittent-demand
methods.

## Setup

Download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition into `data/raw/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python -m forecast.prepare    # validate, clean and write data/processed/sales_prepared.parquet
python -m forecast.eda        # figures in reports/figures and reports/eda_summary.json
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
