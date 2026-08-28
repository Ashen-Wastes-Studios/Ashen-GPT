@echo off
echo Activating virtual environment and running Ashen GPT Trainer...
call cuda\Scripts\activate.bat
python ashen_gpt_trainer.py
pause
