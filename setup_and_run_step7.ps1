# ==============================================================================
# Project 2: Retail Sales & Demand Forecasting
# Step 7 - Tuning & Ensembling (PowerShell Execution Script)
# ==============================================================================

# 1. Navigate to project root and set up virtual environment
cd C:\Projects\retail-demand-forecasting
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

# 2. Allow script activation in current session & activate .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1

# 3. Ensure Step 7 dependencies are installed
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn lightgbm optuna prophet statsmodels plotly joblib pyarrow

# 4. Create output folders for Step 7 artifacts if they do not exist
New-Item -ItemType Directory -Force -Path "src", "models", "reports", "reports/step7", "data/processed", "data/predictions" | Out-Null

# 5. Run Step 7: Optuna LightGBM Tuning, Prophet/ARIMA Blending & Granular Error Analysis
python src/step7_tuning_and_ensembling.py `
    --data-path "data/processed/features_step6.parquet" `
    --stat-preds-path "data/predictions/prophet_arima_val_preds.parquet" `
    --n-trials 30 `
    --horizon 28 `
    --n-splits 3 `
    --output-dir "reports/step7" `
    --model-dir "models"

Write-Host "`n[Step 7 Complete] Tuned model saved in 'models/' and error breakdown reports saved in 'reports/step7/'." -ForegroundColor Green
