
Source Code (Please run in this order):

MassShootingDataPull.py: Script to pull data from a Google Spreadsheet using an API call
MassShootingDataCleanliness.py: Calculates the cleanliness of the data in mass_shootings.csv
movies.py: Script to pull data from the movie database using an API call. ** because of timing restriction you must
    manually run the commented out getMovie#() methods one by one **
MoviesDataCleanliness.py: Calculates the cleanliness of data in all the movie csv files. Appends to badData.txt


Other:

credentials.json: Google credentials for API call in MassShootingDataPull.py
client_id.json: Google ID for API call in MassShootingDataPull.py
badData.txt: Output of the results from MassShootingDataCleanliness.py and MoviesDataCleanliness.py

Data Storage:

mass_shootings.csv: The data from the Google Spreadsheet in a csv format
The below csv files contain the 3200 records of movie data:
    movies.csv
    movies2.csv
    movies3.csv
    movies4.csv

Extra data: Folder that contains data retrieved from other sources not using API for mass shootings.

gun_violence_archive: CSV pulled from Gun Violence Archive
mass_shooting_tracker_XXXX: CSV pulled from Mass Shooting Tracker for corresponding year.