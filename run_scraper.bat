@echo off
echo Activating virtual environment and downloading public domain training data...
call cuda\Scripts\activate.bat
python scrape_public_domain.py
pause
