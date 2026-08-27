@echo off
echo Activating virtual environment and streaming OpenWebText into train_split.txt...
call cuda\Scripts\activate.bat
python scrape_openwebtext.py
pause
