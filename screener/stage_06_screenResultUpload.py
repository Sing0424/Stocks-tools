# stage_06_screenResultUpload.py

from __future__ import print_function
import os
import sys
import datetime
import warnings
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

def upload_results():
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    if not os.path.exists(Config.EXCEL_REPORT_FILE):
        print(f"Error: Report file not found at {Config.EXCEL_REPORT_FILE}.")
        return False

    if not os.path.exists(Config.CREDENTIAL) and not os.path.exists(Config.TOKEN):
        print("Google Drive credentials not found (credentials.json or token.json missing). Skipping upload.")
        return True

    creds = None
    if os.path.exists(Config.TOKEN):
        creds = Credentials.from_authorized_user_file(Config.TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(Config.CREDENTIAL):
                print(f"Error: Credentials file not found at {Config.CREDENTIAL}.")
                return False
            flow = InstalledAppFlow.from_client_secrets_file(Config.CREDENTIAL, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(Config.TOKEN, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        folder_metadata = {
            'name': 'Screen Result',
            'mimeType': 'application/vnd.google-apps.folder'
        }

        folder = service.files().update(
            body=folder_metadata,
            fields='id',
            fileId=Config.GDRIVE_FOLDER_ID
        ).execute()
        folder_id = folder.get("id")
        print(f'Folder ID: "{folder_id}".')

        file_metadata = {
            'name': f'Final_Report_{datetime.date.today()}',
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }
        media = MediaFileUpload(
            Config.EXCEL_REPORT_FILE,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            resumable=True
        )
        file = service.files().update(
            body=file_metadata,
            media_body=media,
            fields='id',
            fileId=Config.GDRIVE_FILE_ID
        ).execute()
        print(f'Uploaded File ID: "{file.get("id")}".')

        return True

    except HttpError as error:
        print(f'Google Drive API error: {error}')
        return False
    except Exception as e:
        print(f'Error uploading to Google Drive: {e}')
        return False

if __name__ == '__main__':
    upload_results()
