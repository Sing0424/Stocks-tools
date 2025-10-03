# stage_06_screenResultUpload.py

from __future__ import print_function
import os.path
import datetime
import warnings
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from config import Config

def upload_results():
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    if os.path.exists(Config.TOKEN):
        creds = Credentials.from_authorized_user_file(Config.TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                Config.CREDENTIAL, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(Config.TOKEN, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        folder_metadata = {
            'name': 'Screen Result',
            'mimeType': 'application/vnd.google-apps.folder'
        }

        folder = service.files().update(body=folder_metadata, fields='id', fileId='1XJmBN164biI7oE3c_ZuQp7UHn7pa3tiG'
                                      ).execute()
        print(F'Folder ID: "{folder.get("id")}".')

        file_metadata = {
            'name': f'Screen_Result_{datetime.date.today()}',
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            "removeParents": [f'{folder.get("id")}'],
            "addParents": [f'{folder.get("id")}']
        }
        media = MediaFileUpload(Config.FINAL_RESULTS_FILE, mimetype=None,
                                resumable=True)
        file = service.files().update(body=file_metadata, media_body=media,
                                      fields='id', fileId='1xHoV8EW40ziRAud57N28kOlw_G_RimpYUpN5LH8sVNs').execute()
        print(F'File ID: "{file.get("id")}".')

        return True

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None
        return False

if __name__ == '__main__':
    upload_results()
