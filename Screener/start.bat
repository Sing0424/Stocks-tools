@echo off

pip install -r requirements.txt

python get_ticker_info.py

python rs_rating_calc.py

python StockScreener.py

pause
