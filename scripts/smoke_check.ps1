$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$argsList = @("scripts\smoke_check.py")
& $python @argsList
exit $LASTEXITCODE
