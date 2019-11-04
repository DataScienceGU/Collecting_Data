import glob
import pandas as pd
import pylab as pl
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
import requests

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def load_gun_sheets():
    path = 'GunViolenceData'
    filenames = glob.glob(path+"/*.csv")
    filenames.sort()

    print(filenames)

    dataframes = []

    for filename in filenames:
        df = pd.read_csv(filename)
        dataframes.append(df)

    all_shooting_data = pd.concat(dataframes, ignore_index=True)
    all_shooting_data.head()

    return all_shooting_data

def getGenres():
    genreDF = pd.DataFrame()
    baseURL = "https://api.themoviedb.org/3/genre/movie/list"
    queryString = {'api_key': 'ddb02b46eef4bf31dc4e8bcbc0222a68',
                   'language': 'en-US'}
    response = requests.get(baseURL, queryString)
    df = response.json()

    genres = df['genres']
    genreDF = genreDF.append(genres, ignore_index=True)

    genreDF.to_csv("movie_genres.csv")
    print(genreDF)

def plot_shooting_histograms(shooting_data):
    #using date as last histogram value from this database, separating month over month per year
    #i figured day by day was too granular.
    shooting_data['date'] = shooting_data['date'].astype('str')
    print(shooting_data['date'].dtype)
    month_year = []
    for date in shooting_data['date']:
        datelist = date.split('/')
        month_year.append((datelist[0] + '/' + datelist[2]))

    print(month_year)
    shooting_data['month_year'] = month_year
    # shooting_data['month_year'] = shooting_data['month_year'].astype('category').cat.codes

    print(shooting_data)
    # shooting_data.hist()
    shooting_data["month_year"].hist()
    plt.show()
    plt.clf()
    plt.close()

    name = "killed"
    shooting_data["killed"].hist()
    pl.suptitle("# Killed per Shooting")
    plt.savefig(name)
    plt.clf()
    plt.close()

    name = "wounded"
    shooting_data["wounded"].hist()
    pl.suptitle("# Wounded per Shooting")
    plt.savefig(name)
    plt.clf()
    plt.close()

    name = "frequency"
    shooting_data["month_year"].hist()
    pl.suptitle("# of Shootings per Month")
    plt.savefig(name)
    plt.clf()
    plt.close()

def plot_movie_histograms(movie_data):
    # Find the correlation between all the pairs of these quantity variables.
    # Include a table of the output in your report, and explain your findings – what does this indicate about your data?
    # Use scatterplots to display the results. Ideally, create a set of scatterplot subplots.
    keywords = []
    violent_words = ['gun', 'shoot', 'murder']
    for movie in movie_data['overview']:
        # print(movie)
        words = []
        for word in violent_words:
            if word in movie:
                words.extend(word)

        keywords.append(words)

    movie_data['keywords'] = keywords
    movie_data['keywords'].value_counts().plot(kind='bar')
    plt.show()




def main():
    shooting_data = load_gun_sheets()
    movie_data = pd.read_csv("all_movies.csv")


    print(shooting_data)
    print("killed max: ")
    print(shooting_data['killed'].max())
    print("wounded max: ")

    print(shooting_data['wounded'].max())
    plot_shooting_histograms(shooting_data)
    plot_movie_histograms(movie_data)



if __name__ == "__main__":
    main()