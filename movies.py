import json
import requests
import csv
import pandas as pd
import numpy as np
import pprint
import time

def getMovies1(baseURL):
    finalDF = pd.DataFrame()

    i = 1977
    for i in range(1977, 2017):
        querystring = {'api_key':'ddb02b46eef4bf31dc4e8bcbc0222a68',
                    'primary_release_year':str(i),
                    'sort_by':'revenue.desc',
                    'language':'en-US',
                    'page':'1'
                    }

        response = requests.get(baseURL, querystring)
        df = response.json()

        films = df['results']
        finalDF = finalDF.append(films, ignore_index=True)


    finalDF.to_csv('movies1.csv')
    print(finalDF)

def getMovies2(baseURL):
    finalDF = pd.DataFrame()

    i = 1977
    for i in range(1977, 2017):
        querystring = {'api_key':'ddb02b46eef4bf31dc4e8bcbc0222a68',
                    'primary_release_year':str(i),
                    'sort_by':'revenue.desc',
                    'language':'en-US',
                    'page':'2'
                    }

        response = requests.get(baseURL, querystring)
        df = response.json()

        films = df['results']
        finalDF = finalDF.append(films, ignore_index=True)


    finalDF.to_csv('movies2.csv')
    print(finalDF)


def getMovies3(baseURL):
    finalDF = pd.DataFrame()

    i = 1977
    for i in range(1977, 2017):
        querystring = {'api_key':'ddb02b46eef4bf31dc4e8bcbc0222a68',
                    'primary_release_year':str(i),
                    'sort_by':'revenue.desc',
                    'language':'en-US',
                    'page':'3'
                    }

        response = requests.get(baseURL, querystring)
        df = response.json()

        films = df['results']
        finalDF = finalDF.append(films, ignore_index=True)


    finalDF.to_csv('movies3.csv')
    print(finalDF)


def getMovies4(baseURL):
    finalDF = pd.DataFrame()

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


    finalDF.to_csv('movies3.csv')
    print(finalDF)



def getGenres():
    genreDF = pd.DataFrame()
    baseURL = "https://api.themoviedb.org/3/genre/movie/list"
    queryString = {'api_key': 'ddb02b46eef4bf31dc4e8bcbc0222a68',
                   'language':'en-US'}
    response = requests.get(baseURL, queryString)
    df = response.json()

    genres = df['genres']
    genreDF = genreDF.append(genres, ignore_index=True)

    genreDF.to_csv("movie_genres.csv")
    print(genreDF)

def main():

    # the getMovies() for loop api call has to be hardcoded to be run separate times
    # each time, the value of page was incremented by 1 in order to work around the API request call time limit
    # and retrieve the minimum number of records (3000) the csv files are then separated, each containing 20 films in
    # order of descending revenue from 1977 to 2016 this year limit was made again to work around the API request call
    # limit the api request limits the records given to 20 per call. So, for example, movies.csv contains the top 20
    # films in order of descending revenue from 1977 to 2016 while movies2.csv contains movies 21-40 in order of descending
    # revenue from 1977 to 2016 and so on and so forth for movies3.csv and movies4.csv. In order to run please comment
    # and uncomment the following methods in order as necessary to run the data:
    baseURL = "https://api.themoviedb.org/3/discover/movie"
    getMovies1(baseURL)
    # getMovies2(baseURL)
    # getMovies3(baseURL)
    # getMovies4(baseURL)

    #this method gets the text name of the genres that the genre ids in the movie csv refer to:
    getGenres()

if __name__ == "__main__":
    main()