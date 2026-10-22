@echo off
title SIGNOVA - Master Launcher
cd /d F:\SIGNOVA
cls
echo ===================================================================
echo                     SIGNOVA ISL TRANSLATOR
echo ===================================================================
echo.
echo   [1] Launch Interactive Web Dashboard GUI (Recommended)
echo       - Web browser interface with Live Studio and Sign Dictionary
echo.
echo   [2] Launch High-Performance Desktop Window
echo       - Native 60 FPS OpenCV HUD window
echo.
echo ===================================================================
set /p choice="Select an option [1 or 2] (Default is 1): "

if "%choice%"=="2" (
    echo Starting Desktop Window...
    call run_webcam.bat
) else (
    echo Starting Web Dashboard GUI in your browser...
    call run_dashboard.bat
)
