# AGENTS.md

## Execution Commands

- **Run Full Pipeline**: `python screener/run_all_stages.py` (or `start.bat`)
- **Run Individual Stages** (must execute in sequence):
  - Stage 1 (Download Symbols): `python screener/stage_01_download_symbols.py`
  - Stage 2 (Filter Symbols): `python screener/stage_02_filter_symbols.py`
  - Stage 3 (Download Prices): `python screener/stage_03_download_price_data.py`
  - Stage 4 (Technical Analysis & Ranking): `python screener/stage_04_technical_analysis_and_ranking.py`
  - Stage 5 (Market Breadth & Sector): `python screener/stage_05_market_breadth.py`
  - Stage 6 (Upload to Google Drive): `python screener/stage_06_screenResultUpload.py`
- **Dependencies**: `pip install -r requirements.txt`

## Operational Quirks & Context

- **Working Directory**: Always run scripts from the project root (`D:\repo\Stocks-tools`). Relative paths (`./data`, `./GoogleAPI`) in `screener/config.py` assume the working directory is the repo root.
- **Environment Setup (`.env`)**:
  - `ALPHA_VANTAGE_API_KEY`: Required for Stage 1.
  - `TG_BOT_TOKEN`: Telegram bot token for Stage 4 alerts.
  - `TG_CHAT_ID`: Telegram chat ID (must be numeric/parseable as `int`).
- **Google Credentials**: Stage 6 requires `GoogleAPI/credentials.json` and `GoogleAPI/token.json`.
- **Pipeline Dependencies**: Stage 5 appends sheets to `data/final_report.xlsx` using `openpyxl`. Stage 4 must run first to create the workbook.
- **Cache Flags**: In `screener/config.py`, `FORCE_REFRESH_SYMBOLS`, `FORCE_REFRESH_FILTERS`, and `FORCE_REFRESH_PRICE_DATA` control whether stages skip execution if outputs already exist in `./data/`. Set to `False` when debugging downstream stages.
- **Business Logic Constraints**:
  - Stage 4 requires at least 252 trading days of data (`PRICE_DATA_PERIOD = "256d"`).
  - Stage 4 automatically excludes stocks from the `Biotechnology` industry.
- **Hardcoded Upload IDs**: Stage 6 updates a specific Google Drive folder ID (`1XJmBN164biI7oE3c_ZuQp7UHn7pa3tiG`) and file ID (`1xHoV8EW40ziRAud57N28kOlw_G_RimpYUpN5LH8sVNs`).
