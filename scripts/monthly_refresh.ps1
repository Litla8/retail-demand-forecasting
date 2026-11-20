# Monthly refresh: archive the current forecast, rebuild the data and forecasts, check earlier forecasts, refresh the dashboard.
# Run from the project root. Schedule it with Windows Task Scheduler (see docs/operations.md).
# Add -Retune to also re-run the hyperparameter search (suggested once a quarter).
param([switch]$Retune)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. .\.venv\Scripts\Activate.ps1

function Step($name, $command) {
    Write-Host "== $name"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

Step "Archive the current forecast" "python -m forecast.monitor --archive"
# Place the refreshed train.csv (history including the newest days) in data\raw before this point.
Step "Regenerate the horizon files" "python -m forecast.horizon"
Step "Prepare the data" "python -m forecast.prepare"
if ($Retune) { Step "Re-tune the model" "python -m forecast.run_tuning" }
Step "Retrain, forecast and plan" "python -m forecast.run_business"

Write-Host "== Check earlier forecasts against the new actuals"
python -m forecast.monitor
$alert = ($LASTEXITCODE -eq 2)
Step "Refresh the dashboard data" "python -m forecast.dashboard_data"
if ($alert) { Write-Warning "Monitoring raised an alert: see reports\monitoring_log.csv"; exit 2 }
Write-Host "Monthly refresh finished without alerts."
