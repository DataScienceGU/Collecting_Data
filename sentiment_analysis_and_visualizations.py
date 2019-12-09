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
import plotly.graph_objs as go
import plotly
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
from plotly.subplots import make_subplots
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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

def format_shooting_data(shooting_data):
    #using date as last histogram value from this database, separating month over month per year
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

def format_movie_data(movie_data):
    #counts frequencies of violent words
    violent_words_frequency= []
    peaceful_words_frequency = []
    violence_rating = []

    #list of words to look for in overview
    violent_words = ["gun", "shoot", "murder", "war", "kill", "pistol", "massacre", "rampage", "violent", "hunt", "mafia"
                     , "attack", "police", "assassin", "crime", "death", "assault", "lacerate", "ravage", "decapacitate",
                     "artillery", "assassinate", "blood", "pillage", "robbery", "annihilate", "battle", "combat", "damage",
                     "deadly", "destroy", "devastation", "domination", "explode", "explosion", "infanticide", "militia",
                     "militant", "maim", "malicious", "rage", "shot", "slaughter", "vicious", "rape"]

    peaceful_words = ["amicable", "peace", "harmony", "ease", "placid", "quiet", "tranquil", "friendly", "mellow", "calm",
                      "pacify", "pacifist", "placate", "serene", "halcyon", "untroubled", "gentle", "restful", "composed",
                      "relaxed", "nice", "kind", "soothing", "easygoing", "happy", "relaxing", "bliss", "carefree",
                      "idyllic", "pleased", "joy", "harmonious", "camaraderie", "merry", "holy", "stability", "respite",
                      "concord", "blessed", "gratitude", "grateful", "balanced", "genuine", "forgiving", "sincere",
                      "uplifted", "radiant", "smiling", "smile", "open-minded", "support", "trusting", "meditate", "meditative",
                      "laughing", "heal"]

    #will hold truth value if violent or not.
    contains_violence = []

    for movie in movie_data['overview']:
        violent_frequency = 0
        peaceful_frequency = 0
        violent = False
        for word in peaceful_words:
            if word in movie:
                peaceful_frequency +=1
        for word in violent_words:
            if word in movie:
                violent = True
                violent_frequency += 1

        contains_violence.append(violent)
        violent_words_frequency.append(violent_frequency)
        peaceful_words_frequency.append(peaceful_frequency)
        violence_rating.append(0+violent_frequency-peaceful_frequency)


    #loading the frequency of violent words in descriptions, and violent movies overall into 2 diff columns in dataframe
    movie_data['violent_words'] = violent_words_frequency
    movie_data['peaceful_words'] = peaceful_words_frequency
    movie_data['violence_rating'] = violence_rating
    print("Frequency of movies with #s of violent words: ")
    print(movie_data['violent_words'].value_counts())
    print("Frequency of movies with #s of peaceful words: ")
    print(movie_data['peaceful_words'].value_counts())
    print("Number of violent movies: ")
    movie_data['violent'] = contains_violence
    print(movie_data['violent'].value_counts())

    return movie_data

def stacked_histogram(shooting_data, movie_data):
    #returns the month/year and # of violent movies released that month in a DF from 2013-2016 (to match shootings)
    movie_data_trunc = process_movie_date(movie_data)

    #turning frequency of shooting per month into a DF and sorting on date.
    shooting_data_consolidated = pd.DataFrame(list(shooting_data['month_year'].value_counts().items()),
                                              columns = ['date', 'shootings'])
    both_data_frequencies = pd.merge(shooting_data_consolidated, movie_data_trunc, on='date', how="left")

    #some months no violent movies were released, so replace with 0
    both_data_frequencies = both_data_frequencies.fillna(0)
    print(both_data_frequencies)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=both_data_frequencies['date'], y=both_data_frequencies['shootings'], name="# Shootings"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=both_data_frequencies['date'], y=both_data_frequencies['num_violent_movies'], name="# Violent Movies Released"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="The Frequency of Shootings and Releases of Violent Movies"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Month and Year")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>primary</b> Shootings Per Month", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> Violent Movies Released Per Month", secondary_y=True)

    #saving plot to plotly account
    # py.plot(fig, filename='basic-line', auto_open=True)

    fig.show()

    return both_data_frequencies

def visualize_bubble_chart(movie_data):
    bubble_movie_data = movie_data.copy()
    bubble_movie_data = bubble_movie_data.drop(columns = ["video", "adult"])
    bubble_movie_data = bubble_movie_data[bubble_movie_data.vote_count != 0]

    for date in bubble_movie_data['release_date']:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        # Convert datetime object to date object.
        date = date.date()

    # getting rid of data before the shooting data
    # start_date = '2000-1-1'
    # mask=(bubble_movie_data['release_date']>start_date)
    # bubble_movie_data=bubble_movie_data.loc[mask]

    #removing outliers
    remove_outlier(bubble_movie_data, "vote_average")

    hover_text = []
    # adding combined text column for hover text labels:
    for index, row in bubble_movie_data.iterrows():
        hover_text_string = row['original_title'] + " violent words: " + str(row['violent_words'])
        hover_text_string += "\n peaceful words: " + str(row['peaceful_words'])
        hover_text_string += "\n number of votes: " + str(row['vote_count'])
        hover_text.append(hover_text_string)

    bubble_movie_data['hover_text'] = hover_text

    print(bubble_movie_data)
    blayout = go.Layout(title=go.layout.Title(
        text='Popularity and Violence Ratings of Movies Released between 2013-Present',),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Release Date'
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Voter Rating'
                )
            )
        )

    fig = go.Figure(data=[go.Scatter(
        x=bubble_movie_data['release_date'],
        y=bubble_movie_data['vote_average'],
        mode='markers',
        text=bubble_movie_data['hover_text'],
        marker=dict(
            size=bubble_movie_data['vote_count'],
            sizemode='area',
            sizeref=2. * int(max(bubble_movie_data['vote_count'])) / (50. ** 2),
            sizemin=4,
            color=bubble_movie_data['violence_rating'],
            colorbar=dict(title="Violence Rating",
                          titleside="top",
                          tickmode="array",
                          tickvals=[min(bubble_movie_data['violence_rating']),
                                    (sum(bubble_movie_data['violence_rating'])/len(bubble_movie_data['violence_rating'])),
                                    max(bubble_movie_data['violence_rating'])],
                          ticktext=["Peaceful", "Neutral", "Violent"],
                          ticks="outside"
                          )
            # color_continuous_scale=bubble_movie_data['violence_rating']
        )
    )], layout = blayout)


    fig.show()
    # py.plot(fig, filename='bubble-graph', auto_open=True)

# from: https://datascience.stackexchange.com/questions/33632/remove-local-outliers-from-dataframe-using-pandas
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def perform_sentiment_analysis(movie_data):
    q1 = movie_data['violence_rating'].quantile(0.33)
    q3 = movie_data['violence_rating'].quantile(0.66)

    violence_classification = []
    for index, row in movie_data.iterrows():
        if(row['violence_rating']<q1):
            violence_classification.append("peaceful")
        elif(row['violence_rating']>q3):
            violence_classification.append("violent")
        else:
            violence_classification.append("neutral")

    movie_data['violence_classification']=violence_classification
    print(movie_data)
    print(movie_data['violence_classification'].value_counts())
    # movie_data.to_csv("manually_labeled_movies.csv")
    run_naive_bayes(movie_data, movie_data['violence_classification'])

def run_naive_bayes(movies, classes):
    # set of labeled data (X_validate, Y_validate)
    myData = movies.copy()
    myData = myData.drop(columns=["Unnamed: 0", "adult", "genre_ids", "id", "original_language", "original_title", "popularity",
                         "release_date", "title", "video", "overview", "vote_average", "vote_count", "violent"])
    print(myData)
    valueArray = myData.values
    lastColNum = len(list(myData))-1
    X = valueArray[:, 0:lastColNum]
    Y = classes
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    seed = 7
    scoring = 'accuracy'

    ######################################################
    # Use different algorithms to build models
    ######################################################

    # Add each algorithm and its name to the model array
    model= GaussianNB()

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ("GAUSS", cv_results.mean(), cv_results.std())
    print(msg)

def run_manually_labeled_naive_bayes():
    my_Data = pd.read_csv("manually_labeled_movies.csv")
    print("IN MANUALLY LABELED DATA")
    print(my_Data)

    my_Data = my_Data.drop(
        columns=["Unnamed: 0", "Unnamed: 0.1","adult", "genre_ids", "id", "original_language", "original_title", "popularity",
                 "release_date", "title", "video", "overview", "vote_average", "vote_count", "violent"])
    print(my_Data)
    valueArray = my_Data.values
    lastColNum = len(list(my_Data)) - 1
    X = valueArray[:, 0:lastColNum]
    Y = my_Data['violence_rating']
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    seed = 7
    scoring = 'accuracy'

    ######################################################
    # Use different algorithms to build models
    ######################################################

    # Add each algorithm and its name to the model array
    model = GaussianNB()

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ("GAUSS", cv_results.mean(), cv_results.std())
    print(msg)


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

def main():
    #setting plotly credentials so I can save graphs.
    chart_studio.tools.set_credentials_file(username='kjc83', api_key='dOWNynMy40XWBRfo6tSm')
    #loading movie and shooting data into two dataframes
    shooting_data = load_gun_sheets()
    movie_data = pd.read_csv("all_movies.csv")

    #preprocessing: formating the month and year
    format_shooting_data(shooting_data)

    #showing stacked bar chart data for shootings and violent movies released in the same month
    movie_data_unseperated = format_movie_data(movie_data)
    perform_sentiment_analysis(movie_data)
    visualize_bubble_chart(movie_data_unseperated)
    run_manually_labeled_naive_bayes()
    movie_and_shooting = stacked_histogram(shooting_data, movie_data_unseperated)

if __name__ == "__main__":
    main()