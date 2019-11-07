# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import silhouette_score


####################################################################################################
# FORMATTING DATAFRAME
####################################################################################################


def categorize(df):
    # Categorize numerical data for movies.csv

    # What is this column??
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df['adult'] = pd.Categorical(df['adult'])
    df['original_language'] = pd.Categorical(df['original_language'])

    # Extract just the year from the date
    df['release_date'] = pd.DatetimeIndex(df['release_date']).year

    # remove brackets from genre_ids column
    df['genre_ids'] = df['genre_ids'].apply(
        lambda x: (removeLeftAndRightBrackets(x)))

    # split up the genre_ids from lists to separate entries
    df = tidy_split(df, 'genre_ids')

    df['genre_ids'] = pd.Categorical(df['genre_ids'])

    # Change all 'category' format columns to numerical
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df


def removeLeftAndRightBrackets(x):
    x = x.strip('[')
    x = x.strip(']')
    return x


def tidy_split(df, column, sep=',', keep=False):
    """
    takes in a dataframe and a column name consisting of a list. will return the dataframe with
    that column name being separted so that each item in the list has its own row

    method found here: 
    https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726

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


def choose_features(df):
    # Select the columns/features to use
    df = pd.concat([df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                    df['release_date'], df['vote_average'], df['vote_count']],
                   axis=1,
                   keys=[df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                         df['release_date'], df['vote_average'], df['vote_count']])

    return df


def preprocess(df):
    # Normalize data
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    return x_scaled, normalizedDataFrame

####################################################################################################
# PLOTTING
####################################################################################################


def pcaPlotting(df, cluster_labels, num, alg_name):
    # 2D PCA plotting and save to a file
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    pca2D = pca2D.fit(df)
    plot_columns = pca2D.transform(df)

    # This shows how good the PCA performs on this dataset
    explained_variance = pca2D.explained_variance_

    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=cluster_labels)
    plt.title("2D PCA, explained_variance: " + str(explained_variance))

    # Write to file
    plt.savefig("movie_pca_" + alg_name + '_' + str(num) + ".png")

    # Clear plot
    plt.clf()

####################################################################################################
# WARD
####################################################################################################


def agglo(x_scaled, normalizedDataFrame, num, filename):
    # Agglomerative clustering using Ward
    agglo = AgglomerativeClustering(
        n_clusters=num, affinity='euclidean', linkage='ward')

    # Get relevant features
    cluster_labels = agglo.fit_predict(x_scaled)
    n_leaves = agglo.n_leaves_
    n_connected_components = agglo.n_connected_components_

    # Silhuette score
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)

    # Write results to file
    with open(filename, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(num, ""))
        f.write('\n\nn_leaves: ' + format(n_leaves, "") + '\n')
        f.write('\n\nn_connected_components: ' +
                format(n_connected_components, "") + '\n')
        f.write("\n\nThe average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')

    # Plot using 2D PCA
    pcaPlotting(normalizedDataFrame, cluster_labels, num, 'agglo')


####################################################################################################
# KMEANS
####################################################################################################


def kmeans(x_scaled, normalizedDataFrame, num, filename):
    # Kmeans clustering
    kmeans = KMeans(n_clusters=num)

    # Get relevant features
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    centroids = kmeans.cluster_centers_

    # Silhuette score
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)

    # Write results to file
    with open(filename, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(num, ""))
        f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\nThe average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')

    # Plot using 2D PCA
    pcaPlotting(normalizedDataFrame, cluster_labels, num, 'kmeans')


####################################################################################################
# DBSCAN
####################################################################################################

def dbscan(x_scaled, normalizedDataFrame, num, filename):
    # DBSCAN clustering
    dbscan = DBSCAN(min_samples=num).fit(x_scaled)

    # Get relevant features
    cluster_labels = dbscan.fit_predict(normalizedDataFrame)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    # Silhuette score
    silhouette_avg = metrics.silhouette_score(x_scaled, cluster_labels)

    # Write results to file
    with open(filename, 'a') as f:
        f.write("For min_samples = ")
        f.write(format(num, ""))
        f.write("\n\nEstimated number of clusters: " +
                format(n_clusters, "") + '\n')
        f.write('\n\nEstimated number of noise points: ' +
                format(n_noise, "") + '\n')
        f.write("\n\nThe average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')

    # Plot using 2D PCA
    pcaPlotting(normalizedDataFrame, cluster_labels, num, 'dbscan')


####################################################################################################
# MAIN
####################################################################################################


def main():
    # Files to write outputs of clusterings to
    myFile1 = "movies_hierchical.txt"
    myFile2 = "movies_kmeans.txt"
    myFile3 = "movies_dbscan.txt"

    ###########
    # READ DATA
    ###########

    # Dataframe for movies data
    # Thows FileNotFoundError in VS Code, but works in Spyder
    df = pd.read_csv('all_movies.csv', sep=',', encoding='latin1')

    # Categorize numerical data
    df = categorize(df)

    # Drop certain columns
    df = choose_features(df)

    # df = tidy_split(df, 'genre_ids')

    # Normalize data
    x_scaled, normalizedDataFrame = preprocess(df)

    ######
    # WARD
    ######

    agglo(x_scaled, normalizedDataFrame, 2, myFile1)
    agglo(x_scaled, normalizedDataFrame, 7, myFile1)
    agglo(x_scaled, normalizedDataFrame, 18, myFile1)

    ########
    # KMEANS
    ########

    kmeans(x_scaled, normalizedDataFrame, 2, myFile2)
    kmeans(x_scaled, normalizedDataFrame, 7, myFile2)
    kmeans(x_scaled, normalizedDataFrame, 18, myFile2)

    ########
    # DBSCAN
    ########

    dbscan(x_scaled, normalizedDataFrame, 1577, myFile3)
    dbscan(x_scaled, normalizedDataFrame, 450, myFile3)
    dbscan(x_scaled, normalizedDataFrame, 175, myFile3)


if __name__ == "__main__":
    main()
