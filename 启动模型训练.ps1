param(
    [string]$Config = "data_sources/buffer_sources.json",
    [string]$ProcessedDir = "data_processed",
    [string]$RunsDir = "model_runs",
    [string]$OverwriteRunDir = "",
    [switch]$OverwriteLatest,
    [int]$GroupedEvalSeedCount = 30
)

$pythonExe = "D:\Software\Miniconda\envs\Distinguish_TC\python.exe"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "Distinguish_TC Python was not found:" -ForegroundColor Red
    Write-Host $pythonExe -ForegroundColor Red
    exit 1
}

if ($OverwriteLatest -and [string]::IsNullOrWhiteSpace($OverwriteRunDir)) {
    $runsRoot = Join-Path $projectRoot $RunsDir
    if (-not (Test-Path -LiteralPath $runsRoot)) {
        Write-Host "No model_runs directory was found, so there is nothing to overwrite." -ForegroundColor Yellow
        exit 1
    }

    $latestRun = Get-ChildItem -LiteralPath $runsRoot -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1

    if ($null -eq $latestRun) {
        Write-Host "No previous training run was found, so there is nothing to overwrite." -ForegroundColor Yellow
        exit 1
    }

    $OverwriteRunDir = Join-Path $RunsDir $latestRun.Name
    Write-Host "Overwrite latest run:" $OverwriteRunDir -ForegroundColor Yellow
}

$arguments = @(
    ".\run_modeling_pipeline.py",
    "--config", $Config,
    "--processed-dir", $ProcessedDir,
    "--runs-dir", $RunsDir,
    "--grouped-eval-seed-count", $GroupedEvalSeedCount
)

if (-not [string]::IsNullOrWhiteSpace($OverwriteRunDir)) {
    $arguments += @("--overwrite-run-dir", $OverwriteRunDir)
}

Push-Location $projectRoot
try {
    Write-Host "Starting modeling pipeline..." -ForegroundColor Cyan
    & $pythonExe @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "Model training failed with exit code $exitCode." -ForegroundColor Red
        exit $exitCode
    }
    Write-Host "Model training finished successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
