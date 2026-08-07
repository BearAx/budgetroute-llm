[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$PythonCommand = if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-CheckedPython @("-m", "compileall", "-q", "src", "tests")
Invoke-CheckedPython @("-m", "ruff", "format", "--check", ".")
Invoke-CheckedPython @("-m", "ruff", "check", ".")
Invoke-CheckedPython @("-m", "mypy", "src")
Invoke-CheckedPython @("-m", "pytest")
Invoke-CheckedPython @("-m", "budgetroute", "security-check", "--config", "configs/serving/fake.yaml")
Invoke-CheckedPython @("-m", "build")
