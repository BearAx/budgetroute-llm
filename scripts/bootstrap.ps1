[CmdletBinding()]
param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $PythonCommand"
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

Invoke-CheckedCommand -Command $PythonCommand -Arguments @("-m", "venv", ".venv")
Invoke-CheckedCommand -Command .\.venv\Scripts\python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedCommand -Command .\.venv\Scripts\python -Arguments @("-m", "pip", "install", "-e", ".[dev]")

Write-Host "Bootstrap complete. The script cannot persist PowerShell activation."
Write-Host "Next: .\.venv\Scripts\Activate.ps1"
Write-Host "Then: python -m budgetroute demo --config configs/serving/fake.yaml"
