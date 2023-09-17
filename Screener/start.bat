@echo off

start "" /REALTIME /B /W pip install -r "E:\Repo\Stocks-tools\Screener\requirements.txt"

start "" /REALTIME /B /W python "E:\Repo\Stocks-tools\Screener\get_ticker_info.py"

start "" /REALTIME /B /W python "E:\Repo\Stocks-tools\Screener\rs_rating_calc.py"

start "" /REALTIME /B /W python "E:\Repo\Stocks-tools\Screener\StockScreener.py"

start "" /REALTIME /B /W python "E:\Repo\Stocks-tools\Screener\screenResultUpload.py"

pause