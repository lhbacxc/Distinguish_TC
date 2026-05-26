$pythonExe = "D:\Software\Miniconda\envs\Distinguish_TC\python.exe"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path $pythonExe)) {
    Write-Host "Distinguish_TC Python was not found:" -ForegroundColor Red
    Write-Host $pythonExe -ForegroundColor Red
    exit 1
}

Push-Location $projectRoot
try {
    & $pythonExe ".\run_app.py"
}
finally {
    Pop-Location
}
