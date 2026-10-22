@echo off
setlocal
cd /d "%~dp0"
title AffectBridge-UQ - Confirmatory v3-r1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_and_run.ps1"
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" echo Experiment incomplete. Read RUN_LOGS\latest_run.log before retrying.
if "%CODE%"=="0" echo Experiment validated. See ENGINE\outputs.
pause
exit /b %CODE%
