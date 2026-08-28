@echo off
echo Activating virtual environment and running BPE + MoE Trainer...
call cuda\Scripts\activate.bat
python bpe_moe_trainer.py
pause
