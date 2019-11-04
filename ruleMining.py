from apyori import apriori
from apyori import dump_as_json
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
#from mlxtend.frequent_patterns import apriori
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
            x.strip('['))
    )
    myDataFrame['genre_ids'] = myDataFrame['genre_ids'].apply(
        lambda x: (
            x.strip(']'))
    )
    return myDataFrame


def tidy_split(df, column, sep='|', keep=False):
    """

    FROM HERE: https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726


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


def main():
    filename = "all_movies.csv"
    myDataFrame = readData(filename)
    vals = myDataFrame[['genre_ids']].values
    myDataFrame = popularityAsBinary(myDataFrame)
    myDataFrame = myDataFrame[['genre_ids', 'popular_1_or_0']]
    newDataFrame = tidy_split(myDataFrame, 'genre_ids', sep=',')
    newDataFrame = removeBrackets(newDataFrame)
    newDataFrame['genre_ids'] = pd.to_numeric(
        newDataFrame['genre_ids'], errors='coerce')
    # print(newDataFrame.values)
    # newDataFrame = newDataFrame.head(100)
    results = list(apriori(newDataFrame.values, min_support=0.0001))
    print(results)
    # newDataFrame = tidy_split(myDataFrame, 'genre_ids', sep=',')
    # print(myDataFrame)
    # newDataFrame = removeBrackets(newDataFrame)
    # print(newDataFrame)
    # # results = list(apriori(myDataFrame))
    # # print(results)
    # newDataFrame['genre_ids'] = pd.to_numeric(
    #     newDataFrame['genre_ids'], errors='coerce')
    # print(newDataFrame.values)
    # newDataFrame = newDataFrame.head(10)
    # te = TransactionEncoder()
    # te_ary = te.fit(newDataFrame).transform(newDataFrame)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df)
    # results = apriori(df, min_support=0.0, use_colnames=True)
    # results.sort_values(by=['support'])
    # print(results)
    # print("\n\n")
    transactions = [
        ['beer', 'nuts'],
        ['beer', 'cheese'],
    ]
    print("\n\n\n")
    results = list(apriori(transactions))
    print(results)
    # te = TransactionEncoder()
    # te_ary = te.fit(vals).transform(vals)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # results = apriori(df, min_support=0.6, use_colnames=True)
    # print(results)


if __name__ == "__main__":
    main()
