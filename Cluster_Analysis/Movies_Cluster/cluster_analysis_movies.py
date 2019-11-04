# Libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

# Files to write outputs of clusterings to
myFile1 = "movies_kmeans.txt"
myFile2 = "movies_dbscan.txt"
myFile3 = "movies_hierchical.txt"

####################################################################################################
# CATEGORIZE


def categorize(df):
    # Categorize numerical data for movies.csv
    df['adult'] = pd.Categorical(df['adult'])
    df['adult'] = df['adult'].cat.codes

    df['genre_ids'] = pd.Categorical(df['genre_ids'])
    df['genre_ids'] = df['genre_ids'].cat.codes

    df['original_language'] = pd.Categorical(df['original_language'])
    df['original_language'] = df['original_language'].cat.codes

    # Extract just the year from the date
    df['release_date'] = pd.DatetimeIndex(df['release_date']).year

    return df

####################################################################################################
# KMEANS


def kmeans(df, num):
    # Not using all the columns
    df = pd.concat([df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                    df['release_date'], df['vote_average'],
                    df['vote_count']],
                   axis=1,
                   keys=[df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                         df['release_date'], df['vote_average'],
                         df['vote_count']])

    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    k = num
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)

    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)

    centroids = kmeans.cluster_centers_

    # Write results to file
    with open(myFile1, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(k, ""))
        f.write("\n\ncluster_labels: " + format(cluster_labels, "") + '\n')
        f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


####################################################################################################
# DBSCAN


def dbscan(df, num):
    # Not using all the columns
    df = pd.concat([df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                    df['release_date'], df['vote_average'],
                    df['vote_count']],
                   axis=1,
                   keys=[df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                         df['release_date'], df['vote_average'],
                         df['vote_count']])

    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    k = num
    dbscan = DBSCAN(min_samples=k).fit(x_scaled)

    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" %
    #       metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" %
    #       metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels,
    #                                            average_method='arithmetic'))
    silhouette_avg = metrics.silhouette_score(x_scaled, labels)

    # Write results to file
    with open(myFile2, 'a') as f:
        f.write("For min_samples = ")
        f.write(format(k, ""))
        # f.write("\n\ncluster_labels: " + format(cluster_labels, "") + '\n')
        # f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


####################################################################################################
# WARD


def agglo(df, num):
    df = pd.concat([df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                    df['release_date'], df['vote_average'],
                    df['vote_count']],
                   axis=1,
                   keys=[df['adult'], df['genre_ids'], df['original_language'], df['popularity'],
                         df['release_date'], df['vote_average'],
                         df['vote_count']])

    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    k = num
    agglo = AgglomerativeClustering(
        n_clusters=k, affinity='euclidean', linkage='ward')
    agglo.fit(x_scaled)
    labels = agglo.labels_

    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, labels)

    # Write results to file
    with open(myFile3, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(k, ""))
        f.write("\n\ncluster_labels: " + format(labels, "") + '\n')
        # f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


def main():
    ###########
    # READ DATA
    ###########

    # Dataframes for movies data
    # Thows FileNotFoundError in VS Code, but works in Spyder
    df1 = pd.read_csv('movies.csv', sep=',', encoding='latin1')

    df2 = pd.read_csv('movies2.csv', sep=',', encoding='latin1')

    df3 = pd.read_csv('movies3.csv', sep=',', encoding='latin1')

    df4 = pd.read_csv('movies4.csv', sep=',', encoding='latin1')

    # Combine dataframes for extra shooting data
    df1 = df1.append(df2)
    df1 = df1.append(df3)
    df1 = df1.append(df4)

    # Categorize numerical data
    df1 = categorize(df1)

    print(df1)

    ######
    # WARD
    ######

    agglo(df1, 5)
    agglo(df1, 10)
    agglo(df1, 15)

    ########
    # KMEANS
    ########

    # three different values for k for the k-means clustering on the data
    kmeans(df1, 5)
    kmeans(df1, 10)
    kmeans(df1, 20)

    ########
    # DBSCAN
    ########

    dbscan(df1, 5)
    dbscan(df1, 10)
    dbscan(df1, 20)


if __name__ == "__main__":
    main()
