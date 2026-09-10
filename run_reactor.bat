@echo off
echo Activating virtual environment and starting Ashen Reactor...
echo Twitch channel: nicktouey_gaming  ^|  Mic: ON  ^|  Model: ashen_gpt_model exclusive
call cuda\Scripts\activate.bat
python ashen_reactor.py %*
pause
