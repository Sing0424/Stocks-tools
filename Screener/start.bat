@echo off

FOR /F "usebackq tokens=1,2,3 delims=/ " %%A in (`echo %date%`) do (
	SET "day=%%A"
	SET "month=%%B"
	SET "year=%%C"
)

SET date=%year%%month%%day%

start "" /REALTIME /B /W pip install -r "requirements.txt"

start "" /REALTIME /B /W python "get_ticker_info.py"

start "" /REALTIME /B /W python "rs_rating_calc.py"

start "" /REALTIME /B /W python "StockScreener.py"

start "" /REALTIME /B /W python "screenResultUpload.py"