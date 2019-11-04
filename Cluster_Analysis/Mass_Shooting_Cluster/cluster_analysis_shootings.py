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
myFile1 = "mass_shootings_kmeans.txt"
myFile2 = "mass_shooting_tracker_kmeans.txt"
myFile3 = "mass_shootings_dbscan.txt"
myFile4 = "mass_shooting_tracker_dbscan.txt"
myFile5 = "mass_shootings_hierchical.txt"
myFile6 = "mass_shooting_tracker_hierchical.txt"

####################################################################################################
# CATEGORIZE


def categorize1(df):
    # Categorize numerical data for mass_shootings.csv
    df['location1'] = pd.Categorical(df['location1'])
    df['location1'] = df['location1'].cat.codes

    df['location2'] = pd.Categorical(df['location2'])
    df['location2'] = df['location2'].cat.codes

    df['prior_signs_mental_health_issues'] = pd.Categorical(
        df['prior_signs_mental_health_issues'])
    df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].cat.codes

    df['weapons_obtained_legally'] = pd.Categorical(
        df['weapons_obtained_legally'])
    df['weapons_obtained_legally'] = df['weapons_obtained_legally'].cat.codes

    return df


def categorize2(df):
    # Categorize numerical data for all the mass_shooting_tracker_"YEAR".csv combined
    df['state'] = pd.Categorical(df['state'])
    df['state'] = df['state'].cat.codes

    # Extract just the year from the date
    df['date'] = pd.DatetimeIndex(df['date']).year

    return df


####################################################################################################
# KMEANS


def kmeans1(df, num):
    # Not using all the columns
    df = pd.concat([df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                    df['location2'], df['age_of_shooter'],
                    df['weapons_obtained_legally'], df['year']],
                   axis=1,
                   keys=[df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                         df['location2'], df['age_of_shooter'],
                         df['weapons_obtained_legally'], df['year']])

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


def kmeans2(df, num):

    # Not using all the columns
    df = pd.concat([df['date'], df['killed'], df['wounded'], df['state']],
                   axis=1,
                   keys=[df['date'], df['killed'], df['wounded'], df['state']])

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
    with open(myFile2, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(k, ""))
        f.write("\n\ncluster_labels: " + format(cluster_labels, "") + '\n')
        f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


####################################################################################################
# DBSCAN


def dbscan1(df, num):
    # Not using all the columns
    df = pd.concat([df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                    df['location2'], df['age_of_shooter'],
                    df['weapons_obtained_legally'], df['year']],
                   axis=1,
                   keys=[df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                         df['location2'], df['age_of_shooter'],
                         df['weapons_obtained_legally'], df['year']])

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
    with open(myFile3, 'a') as f:
        f.write("For min_samples = ")
        f.write(format(k, ""))
        # f.write("\n\ncluster_labels: " + format(cluster_labels, "") + '\n')
        # f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


def dbscan2(df, num):
    # Not using all the columns
    df = pd.concat([df['date'], df['killed'], df['wounded'], df['state']],
                   axis=1,
                   keys=[df['date'], df['killed'], df['wounded'], df['state']])

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

    silhouette_avg = metrics.silhouette_score(x_scaled, labels)

    # Write results to file
    with open(myFile4, 'a') as f:
        f.write("For min_samples = ")
        f.write(format(k, ""))
        # f.write("\n\ncluster_labels: " + format(cluster_labels, "") + '\n')
        # f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


####################################################################################################
# WARD


def agglo1(df, num):
    df = pd.concat([df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                    df['location2'], df['age_of_shooter'],
                    df['weapons_obtained_legally'], df['year']],
                   axis=1,
                   keys=[df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                         df['location2'], df['age_of_shooter'],
                         df['weapons_obtained_legally'], df['year']])

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
    with open(myFile5, 'a') as f:
        f.write("For n_clusters = ")
        f.write(format(k, ""))
        f.write("\n\ncluster_labels: " + format(labels, "") + '\n')
        # f.write('\n\ncentroids: ' + format(centroids, "") + '\n')
        f.write("\n\n The average silhouette_score is : ")
        f.write(format(silhouette_avg, ""))
        f.write('\n\n*************\n\n')


def agglo2(df, num):
    df = pd.concat([df['date'], df['killed'], df['wounded'], df['state']],
                   axis=1,
                   keys=[df['date'], df['killed'], df['wounded'], df['state']])

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
    with open(myFile6, 'a') as f:
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

    # Dataframe for main shooting data
    df1 = pd.read_csv('mass_shootings.csv', sep=',', encoding='latin1')

    # Dataframes for extra shooting data
    df2 = pd.read_csv('mass_shooting_tracker_2013.csv',
                      sep=',', encoding='latin1')
    temp1 = pd.read_csv('mass_shooting_tracker_2014.csv',
                        sep=',', encoding='latin1')
    temp2 = pd.read_csv('mass_shooting_tracker_2015.csv',
                        sep=',', encoding='latin1')
    temp3 = pd.read_csv('mass_shooting_tracker_2016.csv',
                        sep=',', encoding='latin1')
    temp4 = pd.read_csv('mass_shooting_tracker_2017.csv',
                        sep=',', encoding='latin1')
    temp5 = pd.read_csv('mass_shooting_tracker_2018.csv',
                        sep=',', encoding='latin1')
    temp6 = pd.read_csv('mass_shooting_tracker_2019.csv',
                        sep=',', encoding='latin1')

    # Combine dataframes for extra shooting data
    df2 = df2.append(temp1)
    df2 = df2.append(temp2)
    df2 = df2.append(temp3)
    df2 = df2.append(temp4)
    df2 = df2.append(temp5)
    df2 = df2.append(temp6)

    # print(df2)

    # Categorize numerical data
    df1 = categorize1(df1)
    df2 = categorize2(df2)

    print(df2)

    ######
    # WARD
    ######

    agglo1(df1, 5)
    agglo1(df1, 10)
    agglo1(df1, 15)

    agglo2(df2, 5)
    agglo2(df2, 10)
    agglo2(df2, 15)

    ########
    # KMEANS
    ########

    # three different values for k for the k-means clustering on the data
    kmeans1(df1, 5)
    kmeans1(df1, 10)
    kmeans1(df1, 20)

    kmeans2(df2, 5)
    kmeans2(df2, 10)
    kmeans2(df2, 20)

    ########
    # DBSCAN
    ########

    dbscan1(df1, 5)
    dbscan1(df1, 10)
    dbscan1(df1, 20)

    dbscan2(df2, 100)
    dbscan2(df2, 1000)
    dbscan2(df2, 1500)


if __name__ == "__main__":
    main()
