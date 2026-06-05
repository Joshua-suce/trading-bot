param(
    [switch]$Fast
)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$argsList = @("scripts\smoke_check.py")
if ($Fast) {
    $argsList += "--fast"
}

& $python @argsList
exit $LASTEXITCODE
