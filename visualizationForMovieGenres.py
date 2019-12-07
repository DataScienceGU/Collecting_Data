import pandas as pd
import plotly.graph_objects as go


def readData(filename):
    # read the csv data and return a pandas dataframe
    myDataFrame = pd.read_csv(
        filename, sep=",", encoding='latin1')
    return myDataFrame


def cleanGenres(myDataFrame):
    movieData = tidy_split(myDataFrame, "genre_ids")
    movieData = removeBrackets(movieData)
    return movieData


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


def createGraph(movieData):

    valueCounts = movieData['genre_ids'].value_counts()
    genreIds = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36,
                27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]
    genreNames = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
                  'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

    # labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
    # values = [4500, 2500, 1053, 500]
    genreCounts = []
    for genreId in genreIds:
        genreCounts.append(valueCounts[str(genreId)])

    print(genreCounts)

    fig = go.Figure(data=[go.Pie(labels=genreNames, values=genreCounts)])
    fig.show()


def main():
    filename = "all_movies.csv"
    myDataFrame = readData(filename)
    movieData = cleanGenres(myDataFrame)
    createGraph(movieData)


if __name__ == "__main__":
    main()
