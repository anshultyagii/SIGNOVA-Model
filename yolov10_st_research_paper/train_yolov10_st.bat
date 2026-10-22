@echo off
title Train YOLOv10-ST (Nature Scientific Reports 2025 Architecture)
cd /d %~dp0
echo ===========================================================================
echo  Starting YOLOv10-ST Model Training (Swin Transformer + Mish + 25 Signs)
echo ===========================================================================
python train_yolov10_st.py
pause
