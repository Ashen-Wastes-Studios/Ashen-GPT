@echo off
echo Activating Qwen Finetuner
call cuda\Scripts\activate.bat
python qwen_finetune.py
pause