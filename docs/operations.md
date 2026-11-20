# Operating the forecasting system

## The monthly refresh

`scripts/monthly_refresh.ps1` runs the whole cycle in one command (from the project root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\monthly_refresh.ps1          # monthly
powershell -ExecutionPolicy Bypass -File scripts\monthly_refresh.ps1 -Retune  # quarterly: also re-run the tuning
```

What it does, in order:

1. **Archive** the current forecast to `reports/archive/forecast_intervals_<first day>.csv`.
2. **Horizon files:** `python -m forecast.horizon` regenerates `test.csv` and `sample_submission.csv` for the 90 days
   after the newest history (it leaves them alone if they already fit).
3. **Prepare** the data (`python -m forecast.prepare`), which validates it and rebuilds the prepared table.
4. *(optional)* **Re-tune** the model (`python -m forecast.run_tuning`).
5. **Retrain, forecast and plan** (`python -m forecast.run_business`): refits the selected model on all history,
   writes the 90-day forecast with intervals, the reorder plan and the backtest used by the dashboard.
6. **Monitor** (`python -m forecast.monitor`): compares every archived forecast with the actuals that have arrived since,
   and writes `reports/monitoring_log.csv`. Exit code 2 means an alert.
7. **Refresh the dashboard data** (`python -m forecast.dashboard_data`).

Before step 2, put the refreshed `data/raw/train.csv` (the full history including the newest days, columns
`date, store, item, sales`) in place.

### Alert rules (`MonitorConfig` in `forecast/config.py`)

An archived forecast is judged once at least 14 of its days have actuals. It raises an alert when

- its realised WAPE is more than 20% above the backtest WAPE (10.36% for the selected model), or
- fewer than 80% of actuals fall inside its nominal 95% interval. The floor is deliberately below 95% because the
  backtest showed the intervals run narrow (a nominal 95% behaved like about 88-91%).

On an alert, look at `reports/monitoring_log.csv` first (bias, coverage, which forecast), then at the dashboard's
Accuracy tab. A persistent positive or negative bias usually means growth has changed; re-run with `-Retune`.

## Scheduling

**Windows Task Scheduler** (monthly, 1st of the month at 06:00):

```powershell
schtasks /Create /SC MONTHLY /D 1 /ST 06:00 /TN "RetailForecastRefresh" `
  /TR "powershell -ExecutionPolicy Bypass -File C:\Projects\retail-demand-forecasting\scripts\monthly_refresh.ps1"
schtasks /Run /TN "RetailForecastRefresh"     # run it once now to test
schtasks /Query /TN "RetailForecastRefresh" /V /FO LIST   # see the last result (0 = ok, 2 = monitoring alert)
```

**Linux cron** (same cycle):

```cron
0 6 1 * * cd /opt/retail-demand-forecasting && . .venv/bin/activate && python -m forecast.monitor --archive && python -m forecast.horizon && python -m forecast.prepare && python -m forecast.run_business && (python -m forecast.monitor; python -m forecast.dashboard_data)
```

Airflow, Prefect or any scheduler can run the same commands as tasks; nothing in them is specific to Windows.

## Deploying the dashboard

The dashboard reads only the `dashboard_data/` folder, so it can be deployed without the raw data or the models.

1. Run `python -m forecast.run_business` and `python -m forecast.dashboard_data`, then commit `dashboard_data/`.
2. On Streamlit Community Cloud, create an app from the repository with main file `app/dashboard.py`. The lean
   `app/requirements.txt` next to it is the one Streamlit installs (it takes precedence over the root file). Pin the
   package versions from your environment in that file for repeatable builds.
3. **Data rules:** the bundle contains the last 400 days of the Kaggle sales (needed for the charts). Check the
   competition's data rules before making the repository or the app public; the repository is private by default.

## Limits to know about

- The Kaggle history ends on 2017-12-31 and the test period has no actuals, so the monitor has nothing to compare
  until newer data exists. It is tested on synthetic data, and it starts working on the first refresh with real new rows.
- Costs and lead times in the inventory plan are placeholders (`BusinessConfig`); replace them before using the
  savings figures.
