@echo off

start "" /REALTIME /B /W pip install -r requirements.txt

start "" /REALTIME /B /W python get_ticker_info.py

start "" /REALTIME /B /W python rs_rating_calc.py

start "" /REALTIME /B /W python StockScreener.py

start "" /REALTIME /B /W python screenResultUpload.py

pause
