# Starts the backend (port 8000) and frontend (port 5173) in two windows,
# then opens the app in your browser.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$root\.venv")) {
    Write-Host "First run: creating Python environment..."
    python -m venv "$root\.venv"
    & "$root\.venv\Scripts\pip" install -r "$root\backend\requirements.txt"
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "First run: installing frontend packages..."
    Push-Location "$root\frontend"; npm install; Pop-Location
}

Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; ..\.venv\Scripts\python -m uvicorn app.main:app --port 8000 --reload"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npx vite --port 5173"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"
