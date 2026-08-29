@echo off
echo Activating virtual environment and starting Ashen GPT Web Interface...
call cuda\Scripts\activate.bat
python web_chatbot.py
pause
