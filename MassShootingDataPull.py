import apiclient
import httplib2
import oauth2client
from oauth2client import file
from oauth2client import tools
import re
import requests
import shutil
import urllib.parse

SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
SPREADSHEET_ID = '1b9o6uDO18sLxBqPwl_Gh9bnhW-ev_dABH83M5Vb5L8o'

store = oauth2client.file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    flow = oauth2client.client.flow_from_clientsecrets(
        'client_id.json', SCOPES)
    creds = oauth2client.tools.run_flow(flow, store)

service = apiclient.discovery.build(
    'sheets', 'v4', http=creds.authorize(httplib2.Http()))

result = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
spreadsheetUrl = result['spreadsheetUrl']
exportUrl = re.sub("\/edit$", '/export', spreadsheetUrl)
headers = {
    'Authorization': 'Bearer ' + creds.access_token,
}

for sheet in result['sheets']:
    params = {
        'format': 'csv',
        'gid': sheet['properties']['sheetId'],
    }
    queryParams = urllib.parse.urlencode(params)
    url = exportUrl + '?' + queryParams
    response = requests.get(url, headers=headers)

    filePath = 'mass_shootings.csv'
    with open(filePath, 'wb') as csvFile:
        csvFile.write(response.content)
