@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Project virtual environment is missing. Run scripts\create_venv.ps1 first. 1>&2
    exit /b 1
)

pushd "%PROJECT_ROOT%"
"%PYTHON%" -m src.main %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
