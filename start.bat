@echo off

start "" /B /W pip install -r "requirements.txt"

start "" /B /W python "E:\Repo\Stocks-tools\screener\run_all_stages.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_01_download_symbols.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_02_filter_symbols.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_03_download_price_data.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_04_technical_analysis.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_05_rs_ranking.py"
@REM start "" /B /W python "E:\Repo\Stocks-tools\screener\stage_06_screenResultUpload.py"
start "" /B /W streamlit run "E:\Repo\Stocks-tools\chartviewer\app.py"