import json
import requests
import csv
import pandas as pd
import numpy as np
import pprint

def getRequests():
    baseURL = "https://api.themoviedb.org/3/discover/movie"

    finalDF = pd.DataFrame()
    
    #this for loop api call had to be hardcoded to be run separate times
    #each time, the value of page was incremented by 1
    #this was done in order to work around the API request call limit set by the database
    #and retrieve the minimum number of records (3000)
    #the csv files are then separated
    #each csv file contains 20 films in order of descending revenue from 1977 to 2016
    #this year limit was made again to work around the API request call limit
    #the api request limits the records given to 20 per call
    #so, for example, movies.csv contains the top 20 films in order of descending revenue from 1977 to 2016
    #while movies2.csv contains movies 21-40 in order of descending revenue from 1977 to 2016
    #and so on and so forth for movies3.csv and movies4.csv
    i = 1977
    for i in range(1977, 2017):
        querystring = {'api_key':'ddb02b46eef4bf31dc4e8bcbc0222a68',
                    'primary_release_year':str(i),
                    'sort_by':'revenue.desc',
                    'language':'en-US',
                    'page':'4'
                    }

        response = requests.get(baseURL, querystring)
        df = response.json()

        films = df['results']
        finalDF = finalDF.append(films, ignore_index=True)

    finalDF.to_csv('movies4.csv')
    print(finalDF)

def main():
    getRequests()

if __name__ == "__main__":
    main()