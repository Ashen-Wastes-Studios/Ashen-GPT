@echo off
echo Activating virtual environment and starting Ashen GPT Web Interface...
echo Loading settings from: settings.json (override with --settings "path\to\settings.json" or SETTINGS_PATH env)
call cuda\Scripts\activate.bat
python web_chatbot.py %*
pause
