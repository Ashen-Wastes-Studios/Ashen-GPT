@echo off
echo Activating virtual environment and starting BPE + MoE Chatbot...
call cuda\Scripts\activate.bat
python chatbot.py
pause
