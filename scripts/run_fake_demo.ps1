[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$PythonCommand = if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}
& $PythonCommand -m budgetroute demo --config configs/serving/fake.yaml

