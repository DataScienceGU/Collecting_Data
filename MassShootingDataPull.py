import apiclient
import httplib2
import oauth2client
import re
import requests
import urllib.parse

#####
# Use Google Sheets API with oauth2 to pull data from a Google Spreadsheet
# Referenced: https://developers.google.com/sheets/api/quickstart/python
# & https://developers.google.com/identity/protocols/OAuth2ServiceAccount
# & https://stackoverflow.com/questions/11619805/using-the-google-drive-api-to-download-a-spreadsheet-in-csv-format

SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
SPREADSHEET_ID = '1b9o6uDO18sLxBqPwl_Gh9bnhW-ev_dABH83M5Vb5L8o'

# Get credentials
store = oauth2client.file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    # Prompts download of client id from Google
    flow = oauth2client.client.flow_from_clientsecrets(
        'client_id.json', SCOPES)
    creds = oauth2client.tools.run_flow(flow, store)

# Use Spreadsheets v4 API
service = apiclient.discovery.build(
    'sheets', 'v4', http=creds.authorize(httplib2.Http()))

result = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
spreadsheetUrl = result['spreadsheetUrl']
exportUrl = re.sub("\/edit$", '/export', spreadsheetUrl)
headers = {
    'Authorization': 'Bearer ' + creds.access_token,
}

# Downloads data as csv worksheet by worksheet
# Uses requests.get() with credentials defined above
for sheet in result['sheets']:
    # Parameters for query
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
