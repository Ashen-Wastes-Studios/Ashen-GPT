@echo off
echo Activating virtual environment and streaming open-source code into code_train_split.txt...
call cuda\Scripts\activate.bat
python scrape_opensource_code.py
pause
