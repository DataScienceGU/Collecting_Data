from apyori import apriori
from apyori import dump_as_json
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
# from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
import pprint


def readData(filename):
    # read the csv data and return a pandas dataframe
    myDataFrame = pd.read_csv(
        filename, sep=",", encoding='latin1')
    return myDataFrame


def popularityAsBinary(myDataFrame):
    # can change what this popularity metric is, such as 0,1,2
    popularityMedian = myDataFrame['popularity'].median()
    myDataFrame['popular_1_or_0'] = myDataFrame['popularity'].apply(
        lambda x: 0 if x < popularityMedian else 1)
    return myDataFrame


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


def toNumericForGenreIds(myDataFrame):
    # turn the the genre_id data numeric instead of strings
    myDataFrame['genre_ids'] = pd.to_numeric(
        myDataFrame['genre_ids'], errors='coerce')
    return myDataFrame


def prepareForApriori():
    # will run specific functions to prepare the data to run the apriori
    # function on it
    filename = "all_movies.csv"
    myDataFrame = readData(filename)
    myDataFrame = popularityAsBinary(myDataFrame)
    myDataFrame = myDataFrame[['genre_ids', 'popular_1_or_0']]
    newDataFrame = tidy_split(myDataFrame, 'genre_ids', sep=',')
    newDataFrame = removeBrackets(newDataFrame)
    newDataFrame = toNumericForGenreIds(newDataFrame)
    return newDataFrame


def runAprioriAlg(myDataFrame, minSupport):
    # runs the apriori algorithm based on the min_support passed in and
    # returns the result

    # have min_lift equal to a number barely over 1 so that it looks at
    # only itemsets with more than 1 item
    results = list(apriori(myDataFrame.values,
                           min_support=minSupport, min_lift=1.000001))

    print(results)


def main():
    myDataFrame = prepareForApriori()
    runAprioriAlg(myDataFrame, 0.05)


if __name__ == "__main__":
    main()
