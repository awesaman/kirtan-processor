@echo off
REM Unified build script for Windows - runs the Python build script
cd /d %~dp0
python build_windows.py %*

REM Confirm completion
@echo.
@echo Build process completed. Press any key to exit.
pause >nul
