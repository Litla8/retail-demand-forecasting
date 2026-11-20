"""Streamlit dashboard: actual vs forecast, accuracy, inventory plan and monitoring.

Run from the project root:  streamlit run app/dashboard.py
It reads only the bundle in dashboard_data/ (build it with `python -m forecast.dashboard_data`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Streamlit puts app/ on sys.path, not the project root, so add the root to import `forecast`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.dashboard_data import (  # noqa: E402
    Bundle,
    at_risk_table,
    backtest_errors,
    load_bundle,
    model_comparison,
    plot_backtest_series,
    plot_error_bars,
    plot_series_forecast,
    plot_store_item_heatmap,
    series_slice,
)

BUNDLE_DIR = ROOT / "dashboard_data"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Retail demand forecasting", page_icon="📦", layout="wide")


@st.cache_data(show_spinner=False)
def get_bundle(path: str) -> Bundle:
    """Load the dashboard bundle once per session."""
    return load_bundle(Path(path))


def show(fig: plt.Figure) -> None:
    """Render a matplotlib figure and release its memory."""
    st.pyplot(fig)
    plt.close(fig)


def pick_series(bundle: Bundle, key: str) -> Tuple[int, int]:
    """Store and item selectors that share state under ``key``."""
    stores = sorted(bundle.forecast["store"].unique())
    items = sorted(bundle.forecast["item"].unique())
    left, right = st.columns(2)
    store = left.selectbox("Store", stores, key=f"{key}_store")
    item = right.selectbox("Item", items, key=f"{key}_item")
    return int(store), int(item)


if not (BUNDLE_DIR / "summaries.json").is_file():
    st.error("No dashboard data found. Run `python -m forecast.run_business` and `python -m forecast.dashboard_data`, then reload.")
    st.stop()

bundle = get_bundle(str(BUNDLE_DIR))
comparison = model_comparison(bundle.summaries)
business = bundle.summaries["business"]

st.title("📦 Retail demand forecasting")
st.caption("500 store-item series, 90-day forecasts, inventory recommendations")
tab_overview, tab_forecast, tab_accuracy, tab_inventory, tab_monitor = st.tabs(
    ["Overview", "Forecast explorer", "Accuracy", "Inventory plan", "Monitoring"])

with tab_overview:
    chosen = comparison[comparison["selected"]]
    best = chosen.iloc[0] if len(chosen) else comparison.iloc[0]  # the selected model, else the best score
    saving = business["inventory_saving_pct_at_target"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected model WAPE", f"{best['mean_wape']:.2f}%", help=str(best["model"]))
    k2.metric("vs 364-day benchmark", f"{best['vs_benchmark_pct']:.1f}% lower error")
    k3.metric("Inventory saved vs benchmark", f"{saving['vs_benchmark']:.1f}%" if saving["vs_benchmark"] is not None else "n/a")
    k4.metric("Inventory saved vs best baseline", f"{saving['vs_best_baseline']:.1f}%" if saving["vs_best_baseline"] is not None else "n/a")
    st.subheader("Model comparison (all 500 series, four validation windows)")
    st.dataframe(comparison.rename(columns={"mean_wape": "mean WAPE (%)", "vs_benchmark_pct": "vs benchmark (%)"}),
                 use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index("model")["mean_wape"])
    st.info("WAPE is total absolute error divided by total units sold. Lower is better. The inventory savings use "
            "placeholder costs; the inventory comparison at equal fill rate does not depend on them.")

with tab_forecast:
    store, item = pick_series(bundle, "explore")
    history, forecast = series_slice(bundle.history, store, item), series_slice(bundle.forecast, store, item)
    show(plot_series_forecast(history, forecast, f"Store {store}, item {item}: forecast with 80% and 95% intervals"))
    past = series_slice(bundle.backtest, store, item).assign(group="series")
    errors, benchmark = backtest_errors(past, "group"), backtest_errors(past, "group", model="benchmark")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model WAPE on past windows", f"{errors['wape'].iloc[0]:.1f}%")
    c2.metric("Benchmark WAPE", f"{benchmark['wape'].iloc[0]:.1f}%")
    c3.metric("Average units per day", f"{history['sales'].mean():.0f}")
    st.subheader("Weekly reorder plan for this series")
    st.dataframe(series_slice(bundle.plan, store, item).round(1), use_container_width=True, hide_index=True)

with tab_accuracy:
    windows = sorted(bundle.backtest["window_end"].unique())
    chosen_window = st.selectbox("Validation window", ["all windows", *windows])
    data = bundle.backtest if chosen_window == "all windows" else bundle.backtest[bundle.backtest["window_end"] == chosen_window]
    c1, c2 = st.columns(2)
    with c1:
        show(plot_error_bars(backtest_errors(data, "store"), "store", "WAPE by store"))
    with c2:
        weekday = backtest_errors(data, "weekday").assign(weekday=lambda t: t["weekday"].map(dict(enumerate(WEEKDAYS))))
        show(plot_error_bars(weekday, "weekday", "WAPE by weekday"))
    st.subheader("WAPE per store and item")
    show(plot_store_item_heatmap(backtest_errors(data, ["store", "item"])))
    st.caption("Error follows series size: the smallest series are the noisiest, which is random variation, not a model weakness.")
    st.subheader("Actual versus forecast in a validation window")
    s_store, s_item = pick_series(bundle, "acc")
    window = st.selectbox("Window to plot", windows, index=len(windows) - 1)
    show(plot_backtest_series(series_slice(bundle.backtest, s_store, s_item), window, f"Store {s_store}, item {s_item}, window ending {window}"))

with tab_inventory:
    opt = business["cost_optimal"]["selected_model"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Service level in the plan", f"{business['plan']['service_level']:.0%}")
    p2.metric("Fill rate in simulation", f"{opt['fill_rate']:.2%}")
    p3.metric("Total forecast units (quarter)", f"{business['plan']['total_forecast_units']:,.0f}")
    st.warning("The intervals run slightly narrow on later data: a nominal 98% safety stock behaved like roughly 94%.")
    st.subheader("Series needing the largest safety buffer")
    st.dataframe(at_risk_table(bundle.plan).round(1), use_container_width=True, hide_index=True)
    st.subheader("Reorder plan")
    f1, f2 = st.columns(2)
    store_filter = f1.multiselect("Stores", sorted(bundle.plan["store"].unique()))
    item_filter = f2.multiselect("Items", sorted(bundle.plan["item"].unique()))
    plan = bundle.plan
    if store_filter:
        plan = plan[plan["store"].isin(store_filter)]
    if item_filter:
        plan = plan[plan["item"].isin(item_filter)]
    st.dataframe(plan.round(1), use_container_width=True, hide_index=True)
    st.download_button("Download plan (CSV)", plan.to_csv(index=False).encode("utf-8"), "reorder_plan.csv", "text/csv")

with tab_monitor:
    st.write("Each month the job archives the current forecast, refreshes the data, retrains and compares the archived "
             "forecast with the actuals that have arrived. An alert is raised when the realised error is more than 20% above "
             "the backtest error, or when the 95% interval covers less than 80% of actuals.")
    if bundle.monitoring.empty:
        st.info("No archived forecast has enough realised days yet, so there is nothing to compare. Results appear here after "
                "the first monthly run with newer data.")
    else:
        st.dataframe(bundle.monitoring, use_container_width=True, hide_index=True)
