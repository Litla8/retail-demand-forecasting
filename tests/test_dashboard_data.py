"""Tests for forecast.dashboard_data (no Streamlit needed)."""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forecast.baselines import build_baselines
from forecast.business import interval_table, reorder_plan
from forecast.config import BaselineConfig, ForecastConfig
from forecast.dashboard_data import (
    at_risk_table,
    backtest_errors,
    export_bundle,
    load_bundle,
    read_monitoring,
    model_comparison,
    plot_backtest_series,
    plot_error_bars,
    plot_series_forecast,
    plot_store_item_heatmap,
    save_backtest,
    series_slice,
)
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.splits import make_validation_windows
from forecast.wide import to_wide, window_arrays

SUMMARIES = {
    "baseline": {"baselines": {"seasonal_naive_364": {"mean_wape": 14.74}, "month_weekday_profile_multi_year": {"mean_wape": 11.01}}},
    "step7": {"variants_mean_wape": {"gbm_default": 10.455, "gbm_tuned_calibrated": 10.355, "gbm_tuned_calibrated_blend": 10.355},
              "selected_variant": "gbm_tuned_calibrated"},
    "business": {"inventory_saving_pct_at_target": {"vs_benchmark": 5.8, "vs_best_baseline": 3.2}},
}


@pytest.fixture
def pipeline_outputs(raw_frames: RawFrames, cfg: ForecastConfig):
    """A prepared table plus every file the pipeline would have written before the export."""
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    wide = to_wide(prepared, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    forecasters = build_baselines(replace(BaselineConfig(), names=(cfg.benchmark, "month_weekday_profile_multi_year")))
    records = {name: [] for name in (*forecasters, "selected_model")}
    for window in windows:
        arrays = window_arrays(wide, window)
        for name, forecaster in forecasters.items():
            records[name].append((arrays.actual, forecaster(arrays.history, arrays.history_dates, arrays.future_dates)))
        records["selected_model"].append(records["month_weekday_profile_multi_year"][-1])
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    save_backtest(records, wide, windows, cfg.paths.backtest_forecasts)
    dates = pd.date_range(wide.dates[-1] + pd.Timedelta(days=1), periods=cfg.horizon_days)
    forecast = np.full((wide.n_series, cfg.horizon_days), 40.0)
    errors = np.random.default_rng(42).normal(0, 1, 2000)
    interval_table(forecast, wide.series, dates, errors, (0.8, 0.95)).to_csv(cfg.paths.forecast_intervals, index=False)
    reorder_plan(forecast, wide.series, dates, 1.5, 0.95, cfg.business).to_csv(cfg.paths.reorder_plan, index=False)
    for path, content in ((cfg.paths.business_summary, SUMMARIES["business"]), (cfg.paths.step7_summary, SUMMARIES["step7"]),
                          (cfg.paths.baseline_summary, SUMMARIES["baseline"])):
        path.write_text(json.dumps(content))
    return prepared, wide, windows


def test_backtest_export_layout(pipeline_outputs, cfg: ForecastConfig) -> None:
    _, wide, windows = pipeline_outputs
    backtest = pd.read_csv(cfg.paths.backtest_forecasts, parse_dates=["date"])
    assert len(backtest) == len(windows) * wide.n_series * cfg.horizon_days
    assert {"window_end", "store", "item", "date", "horizon_day", "actual", "benchmark", "baseline", "model"} <= set(backtest.columns)
    assert backtest["horizon_day"].min() == 1 and backtest["horizon_day"].max() == cfg.horizon_days
    first = backtest[backtest["window_end"] == windows[0].valid_end.date().isoformat()]
    assert first["date"].min() == windows[0].valid_start and first["actual"].notna().all()


def test_export_and_load_bundle_round_trip(pipeline_outputs, cfg: ForecastConfig, tmp_path) -> None:
    prepared, wide, _ = pipeline_outputs
    out = export_bundle(cfg, prepared, out_dir=tmp_path / "bundle")
    bundle = load_bundle(out)
    assert len(bundle.history) == wide.n_series * 400 and bundle.history["date"].max() == wide.dates[-1]
    assert len(bundle.forecast) == wide.n_series * cfg.horizon_days and "lower_95" in bundle.forecast
    assert bundle.plan["order_date"].min() > bundle.history["date"].max() and bundle.monitoring.empty
    assert set(bundle.summaries) == {"baseline", "step7", "business"}
    assert len(series_slice(bundle.forecast, 1, 2)) == cfg.horizon_days


def test_bundle_errors_name_the_missing_step(pipeline_outputs, cfg: ForecastConfig, tmp_path) -> None:
    prepared, _, _ = pipeline_outputs
    cfg.paths.business_summary.unlink()
    with pytest.raises(FileNotFoundError, match="run_business"):
        export_bundle(cfg, prepared, out_dir=tmp_path / "bundle")
    with pytest.raises(FileNotFoundError, match="dashboard_data"):
        load_bundle(tmp_path / "nothing_here")


def test_model_comparison_orders_models_and_computes_the_improvement() -> None:
    table = model_comparison(SUMMARIES)
    assert table["mean_wape"].is_monotonic_increasing
    best = table.iloc[0]
    assert best["vs_benchmark_pct"] == pytest.approx(100 * (1 - 10.355 / 14.74), abs=0.05)
    assert table.loc[table["model"] == "seasonal_naive_364", "vs_benchmark_pct"].iloc[0] == 0.0
    assert table["model"].tolist()[:2] == ["gbm_tuned_calibrated", "gbm_tuned_calibrated_blend"]  # a tie: simpler first
    assert table["selected"].tolist().count(True) == 1 and bool(table.loc[0, "selected"])


def test_backtest_errors_are_computed_from_sums() -> None:
    backtest = pd.DataFrame({
        "store": [1, 1, 2, 2], "item": [1, 1, 1, 1],
        "date": pd.to_datetime(["2017-01-02", "2017-01-03", "2017-01-02", "2017-01-03"]),
        "actual": [10.0, 30.0, 20.0, 20.0], "model": [12.0, 30.0, 20.0, 10.0], "benchmark": [10.0, 30.0, 20.0, 20.0]})
    by_store = backtest_errors(backtest, "store").set_index("store")
    assert by_store.loc[1, "wape"] == pytest.approx(100 * 2 / 40) and by_store.loc[2, "bias"] == pytest.approx(-25.0)
    assert backtest_errors(backtest, "store", model="benchmark")["wape"].eq(0).all()
    weekday = backtest_errors(backtest, "weekday").set_index("weekday")
    assert set(weekday.index) == {0, 1}  # 2017-01-02 is a Monday


def test_at_risk_table_ranks_by_relative_buffer() -> None:
    plan = pd.DataFrame({"store": [1, 2, 3], "item": [1, 1, 1], "order_date": pd.to_datetime(["2018-01-01"] * 3),
                         "forecast_units": [1000.0, 100.0, 400.0], "safety_stock": [50.0, 30.0, 40.0],
                         "order_up_to": [1050.0, 130.0, 440.0], "service_level": 0.95})
    top = at_risk_table(plan, top=2)
    assert top["store"].tolist() == [2, 3] and top["buffer_pct"].round(1).tolist() == [30.0, 10.0]


def test_chart_functions_return_figures(pipeline_outputs, cfg: ForecastConfig, tmp_path) -> None:
    prepared, _, windows = pipeline_outputs
    bundle = load_bundle(export_bundle(cfg, prepared, out_dir=tmp_path / "bundle"))
    figures = [
        plot_series_forecast(series_slice(bundle.history, 1, 1), series_slice(bundle.forecast, 1, 1), "title"),
        plot_backtest_series(series_slice(bundle.backtest, 1, 1), windows[0].valid_end.date().isoformat(), "title"),
        plot_error_bars(backtest_errors(bundle.backtest, "store"), "store", "title"),
        plot_store_item_heatmap(backtest_errors(bundle.backtest, ["store", "item"])),
    ]
    assert all(isinstance(fig, plt.Figure) for fig in figures)
    assert len(figures[0].axes[0].get_lines()) == 2 and figures[1].axes[0].get_title() == "title"
    for fig in figures:
        plt.close(fig)


def test_read_monitoring_tolerates_missing_empty_and_header_only_logs(tmp_path) -> None:
    assert read_monitoring(tmp_path / "missing.csv").empty
    empty = tmp_path / "empty.csv"
    empty.write_text("")  # what an old run wrote when nothing could be judged yet
    assert read_monitoring(empty).empty
    header_only = tmp_path / "header.csv"
    header_only.write_text("archive,wape\n")
    frame = read_monitoring(header_only)
    assert frame.empty and list(frame.columns) == ["archive", "wape"]
