@echo off
title SIGNOVA - Indian Sign Language Web Dashboard
cd /d F:\SIGNOVA
echo ========================================================
echo Starting SIGNOVA Streamlit Web Dashboard...
echo Opening browser at http://localhost:8501 ...
echo ========================================================
start "" http://localhost:8501
py -3.12 -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
pause
