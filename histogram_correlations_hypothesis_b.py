import glob
import pandas as pd
import pylab as pl
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import requests
import ast

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def load_gun_sheets():
    path = 'GunViolenceData'
    filenames = glob.glob(path+"/*.csv")
    filenames.sort()

    dataframes = []

    for filename in filenames:
        df = pd.read_csv(filename)
        dataframes.append(df)

    all_shooting_data = pd.concat(dataframes, ignore_index=True)
    all_shooting_data.head()

    return all_shooting_data

def get_genres():
    genreDF = pd.DataFrame()
    baseURL = "https://api.themoviedb.org/3/genre/movie/list"
    queryString = {'api_key': 'ddb02b46eef4bf31dc4e8bcbc0222a68',
                   'language': 'en-US'}
    response = requests.get(baseURL, queryString)
    df = response.json()

    genres = df['genres']
    genreDF = genreDF.append(genres, ignore_index=True)

    genreDF.to_csv("movie_genres.csv")
    print("Index of Genres:")
    print(genreDF)
    return genreDF

def shooting_histograms(shooting_data):
    #using date as last histogram value from this database, separating month over month per year
    #i figured day by day was too granular.
    #Also formatted it to be the same as the dates in the release date column in movie data base.
    shooting_data['date'] = shooting_data['date'].astype('str')

    month_year = []
    for date in shooting_data['date']:
        month = 0;
        year = 0;
        datelist = date.split('/')
        if datelist[2] == "18":
            year = "2018"
        elif datelist[2] == "19":
            year = "2019"
        else:
           year = datelist[2]

        if len(datelist[0]) ==1:
            month = "0" + datelist[0]
        else:
            month = datelist[0]

        month_year.append(month+"/"+year)

    shooting_data['month_year'] = month_year
    plot_histogram(shooting_data["month_year"], "frequency_shootings", "# of Shootings per Month")
    plot_histogram(shooting_data["killed"], "killed", "# Killed per Shooting")
    plot_histogram(shooting_data["wounded"], "wounded", "# Wounded per Shooting")




def plot_histogram(myData, name, title):
    myData.hist()
    pl.suptitle(title)
    plt.savefig(name)
    plt.clf()
    plt.close()

def movie_histograms(movie_data):
    #counts frequencies of violent words
    violent_words_frequency= []

    #list of words to look for in overview
    violent_words = ["gun", "shoot", "murder", "war", "kill", "pistol", "massacre", "rampage", "violent", "hunt", "mafia"
                     , "attack", "police", "assassin", "crime", "gory", "outrage", "terrorism", "gore", "rage", ]

    #will hold truth value if violent or not.
    contains_violence = []

    for movie in movie_data['overview']:
        frequency = 0
        violent = False
        for word in violent_words:
            if word in movie:
                violent = True
                frequency += 1

        contains_violence.append(violent)
        violent_words_frequency.append(frequency)

    #loading the frequency of violent words in descriptions, and violent movies overall into 2 diff columns in dataframe
    movie_data['violent_words'] = violent_words_frequency
    print("Frequency of movies with #s of violent words: ")
    print(movie_data['violent_words'].value_counts())
    print("Number of violent movies: ")
    movie_data['violent'] = contains_violence
    print(movie_data['violent'].value_counts())

    plot_histogram(movie_data['violent_words'], "violent_words", "# of violent words per movie description")

    #the below code handles processing and plotting movie genre data
    #turning the string of movie ids into a list for processing
    for movie_genre in movie_data['genre_ids']:
        ast.literal_eval(movie_genre)

    #We have to make a new row for each genre in the list of genre ids per record so we can count the frequency
    #so I save and return the original version for other processing.
    copy_unseparated_movies = movie_data

    #splitting the list of genre ids into separate rows to analyze frequency:
    movie_data = tidy_split(movie_data, "genre_ids")
    movie_data = removeBrackets(movie_data)

    #printing to check properly split
    print(movie_data.head(n=5))

    #plotting and showing genre frequencies:
    print("Frequencies of Genres: ")
    print(movie_data['genre_ids'].value_counts())
    plot_histogram(movie_data['genre_ids'], "genres", "Frequencies of Genres")
    return copy_unseparated_movies


def removeBrackets(myDataFrame):
    # remove brackets from the data
    myDataFrame['genre_ids'] = myDataFrame['genre_ids'].apply(
        lambda x: (
            removeLeftAndRightBrackets(x))
    )
    return myDataFrame


def removeLeftAndRightBrackets(x):
    x = x.strip('[')
    x = x.strip(']')
    x = x.strip()
    return x


def tidy_split(df, column, sep=',', keep=False):
    """
    takes in a dataframe and a column name consisting of a list. will return the dataframe with
    that column name being separted so that each item in the list has its own row

    method found here: https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726


    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df

def plot_scatterplots(shooting_data, movie_data):
    #going to plot and correlate three quantitative variables: # violent movies per month, #shootings per month

    #returns the month/year and # of violent movies released that month in a DF from 2013-2016 (to match shootings)
    movie_data_trunc = process_movie_date(movie_data)

    #turning frequency of shooting per month into a DF and sorting on date.
    shooting_data_consolidated = pd.DataFrame(list(shooting_data['month_year'].value_counts().items()),
                                              columns = ['date', 'shootings'])
    both_data_frequencies = pd.merge(shooting_data_consolidated, movie_data_trunc, on='date', how="left")
    del both_data_frequencies['date']

    #some months no violent movies were released, so replace with 0
    both_data_frequencies = both_data_frequencies.fillna(0)
    print(both_data_frequencies)

    scatterplot(both_data_frequencies,'num_violent_movies', 'shootings',
                "Correlation Between # of Violent Movies Released and Shootings in Same Month",
                "# violent movies released in the month", "#shootings in the month","correlation_movies_shootings")
    scatterplot(both_data_frequencies, 'num_movies_in_violent_genres', 'shootings',
                "Correlation Between # of Movies in Sensational Genres Released and Shootings in Same Month",
                "# movies in sensational genres released per month", "#shootings in the month", "correlation_movie_genre_shootings")
    scatterplot(both_data_frequencies, 'num_movies_in_violent_genres', 'num_violent_movies',
                "Correlation Between # of Movies in Sensational Genres Released and # of Violent Genres",
                "# movies in sensational genres released per month", "#violent movies in the month",
                "correlation_movie_genre_violent_movie")

    return both_data_frequencies

def scatterplot(both_data_frequencies, xaxis, yaxis, title, xlabel, ylabel, filename):
    both_data_frequencies.plot(x=xaxis, y=yaxis, style='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()
    plt.clf()
    plt.close()
    print("Pearson correlation coefficient and ptail values for " + xaxis + " and " + yaxis +": ")
    print(stats.pearsonr(both_data_frequencies[xaxis], both_data_frequencies[yaxis]))

def process_movie_date(movie_data):
    month_year=[]

    #formatting the release dates by month and year
    for date in movie_data['release_date']:
        datelist = date.split('-')
        month_year.append((datelist[1] + '/' + datelist[0]))

    movie_data['month_year'] = month_year
    sensational_genre_names = ['Action', 'Adventure', 'Horror', 'Thriller', 'Drama', 'Western', 'Science Fiction']
    sensational_genre_ids = []

    genre_guide = get_genres()
    for index, genre in genre_guide.iterrows():
        if genre['name'] in sensational_genre_names:
            sensational_genre_ids.append(str(genre['id']))

    print("GENRES LOOKING FOR: ")
    print(sensational_genre_ids)

    #pulling out only relevant movies(violent, or belong to a sensational genre) from dataframe:
    violent_movies_per_month = {}
    sensational_genres_per_month = {}
    for index, movie in movie_data.iterrows():
        if movie['violent']:
            if movie['month_year'] not in violent_movies_per_month:
                violent_movies_per_month[movie['month_year']] = 1
            elif movie['month_year'] in violent_movies_per_month:
                violent_movies_per_month[movie['month_year']] += 1
        for genre in sensational_genre_ids:
            if str(genre) in movie['genre_ids']:
                if movie['month_year'] in sensational_genres_per_month:
                    sensational_genres_per_month[movie['month_year']] += 1
                else:
                    sensational_genres_per_month[movie['month_year']] = 1

    print("SENSATIONAL MOVIE FREQUENCY: ")
    print(sensational_genres_per_month)


    remove_movies_before_shootings(violent_movies_per_month)
    remove_movies_before_shootings(sensational_genres_per_month)
    movie_data_violence = pd.DataFrame(list(violent_movies_per_month.items()), columns = ['date', 'num_violent_movies'])
    movie_data_genre = pd.DataFrame(list(sensational_genres_per_month.items()), columns = ['date', 'num_movies_in_violent_genres'])
    movie_data_2013_on =  pd.merge(movie_data_violence, movie_data_genre, on='date', how = 'outer')

    #printing merged data:
    print(movie_data_2013_on)
    return movie_data_2013_on

def remove_movies_before_shootings(violent_movies_per_month):
    #removes movies before the mass shooting.
    list_keys = list(violent_movies_per_month.keys())
    list_keys.sort(key=lambda x: datetime.datetime.strptime(x, '%m/%Y'))
    for key in list_keys:
        if (key=="01/2013"):
            return
        else:
            del violent_movies_per_month[key]

def hypothesis_test_movies_and_shooting(movie_and_shooting):
    #source for linear regression method: https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d

    #convert to numpy array
    X = movie_and_shooting.iloc[:, 1].values.reshape(-1, 1)
    Y = movie_and_shooting.iloc[:, 0].values.reshape(-1, 1)

    #creating model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    print("Linear Model Coefficient: ")
    print(linear_regressor.coef_)


    plt.title("Linear Regression on # Violent Movies and # Shootings in the Same Month")
    plt.xlabel("# Violent Movies released in a month")
    plt.ylabel("# Shootings that month")
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.savefig("linear_regression_violent_movies_shootings")
    plt.show()
    plt.clf()
    plt.close()

def main():
    #loading movie and shooting data into two dataframes
    shooting_data = load_gun_sheets()
    movie_data = pd.read_csv("all_movies.csv")

    #Max values for killed and wounded in shooting history:
    print(shooting_data)
    print("killed max: ")
    print(shooting_data['killed'].max())
    print("wounded max: ")
    print(shooting_data['wounded'].max())

    #show histograms for shooting_data (Num killed per shooting, num wounded per shooting,
    # and frequency of shootings month over month)
    shooting_histograms(shooting_data)

    #show histograms for movie data (# of movies identified to be violent, # of violent words per description,
    # and frequency of movies based on genres
    # genres are encoded by ids given by our data source, which I pull in get_genres
    movie_data_unseperated = movie_histograms(movie_data)
    movie_and_shooting = plot_scatterplots(shooting_data, movie_data_unseperated)
    hypothesis_test_movies_and_shooting(movie_and_shooting)




if __name__ == "__main__":
    main()