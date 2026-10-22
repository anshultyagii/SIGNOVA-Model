@echo off
title YOLOv10-ST Indian Sign Language Translator (Nature 2025)
cd /d %~dp0
echo ========================================================
echo Starting YOLOv10-ST (Swin Transformer + Mish) Live Stream
echo ========================================================
python realtime_webcam.py
pause
