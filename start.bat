@echo off

start "" /B /W pip install -r "requirements.txt"

start "" /B /W python "run_all_stages.py"
@REM start "" /B /W python "stage_01_download_symbols.py"
@REM start "" /B /W python "stage_02_filter_symbols.py"
@REM start "" /B /W python "stage_03_download_price_data.py"
@REM start "" /B /W python "stage_04_technical_analysis.py"
@REM start "" /B /W python "stage_05_rs_ranking.py"
@REM start "" /B /W python "screenResultUpload.py"