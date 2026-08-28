@echo off
echo Generating validation split from train_split.txt...
call cuda\Scripts\activate.bat
python create_validation_split.py
pause
