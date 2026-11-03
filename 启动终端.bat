@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 燃料电池 RUL 预测终端
echo ============================================
echo   燃料电池剩余寿命预测终端（Streamlit）
echo   正在启动，请稍候... 浏览器会自动打开
echo   如需停止：关闭本窗口，或在此按 Ctrl+C
echo ============================================
"C:\Users\YOGA\.conda\envs\pytorch_env\python.exe" -m streamlit run app.py
pause
