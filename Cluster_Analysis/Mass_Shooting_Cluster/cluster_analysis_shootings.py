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

# All methods that end with 1 deal with mass_shootings.csv
# All methods that end with 2 deal with mass_shooting_tracker_year.csv


####################################################################################################
# FORMATTING DATA FRAME
####################################################################################################


def categorize1(df):
    # Categorize numerical data
    df['location1'] = pd.Categorical(df['location1'])
    df['location2'] = pd.Categorical(df['location2'])
    df['prior_signs_mental_health_issues'] = pd.Categorical(
        df['prior_signs_mental_health_issues'])
    df['weapons_obtained_legally'] = pd.Categorical(
        df['weapons_obtained_legally'])

    # Change all 'category' format columns to numerical
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df


def categorize2(df):
    # Categorize numerical data
    df['state'] = pd.Categorical(df['state'])
    df['state'] = df['state'].cat.codes

    # Extract just the year from the date
    df['date'] = pd.DatetimeIndex(df['date']).year

    return df


def choose_features1(df):
    # Select the columns/features to use
    df = pd.concat([df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                    df['location2'], df['age_of_shooter'],
                    df['weapons_obtained_legally'], df['year']],
                   axis=1,
                   keys=[df['location1'], df['fatalities'], df['injured'], df['total_victims'],
                         df['location2'], df['age_of_shooter'],
                         df['weapons_obtained_legally'], df['year']])

    return df


def choose_features2(df):
    # Select the columns/features to use
    df = pd.concat([df['date'], df['killed'], df['wounded'], df['state']],
                   axis=1,
                   keys=[df['date'], df['killed'], df['wounded'], df['state']])

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


def pcaPlotting(df, cluster_labels, num, alg_name, original_or_extra):
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
    plt.savefig("shootings_pca_" + original_or_extra +
                '_' + alg_name + '_' + str(num) + ".png")

    # Clear plot
    plt.clf()

####################################################################################################
# WARD
####################################################################################################


def agglo(x_scaled, normalizedDataFrame, num, filename, original):
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
    # original = flag that indicates which dataset was used
    if (original):
        pcaPlotting(normalizedDataFrame, cluster_labels,
                    num, 'agglo', 'original')
    else:
        pcaPlotting(normalizedDataFrame, cluster_labels, num, 'agglo', 'extra')

####################################################################################################
# KMEANS
####################################################################################################


def kmeans(x_scaled, normalizedDataFrame, num, filename, original):
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
    # original = flag that indicates which dataset was used
    if (original):
        pcaPlotting(normalizedDataFrame, cluster_labels,
                    num, 'kmeans', 'original')
    else:
        pcaPlotting(normalizedDataFrame, cluster_labels,
                    num, 'kmeans', 'extra')

####################################################################################################
# DBSCAN
####################################################################################################


def dbscan(x_scaled, normalizedDataFrame, num, filename, original):
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

    # Flag that indicates which dataset was used
    if (original):
        pcaPlotting(normalizedDataFrame, cluster_labels,
                    num, 'dbscan', 'original')
    else:
        pcaPlotting(normalizedDataFrame, cluster_labels,
                    num, 'dbscan', 'extra')

####################################################################################################
# MAIN
####################################################################################################


def main():
    # Files to write outputs of clusterings to
    myFile1 = "mass_shootings_hierchical.txt"
    myFile2 = "mass_shooting_tracker_hierchical.txt"
    myFile3 = "mass_shootings_kmeans.txt"
    myFile4 = "mass_shooting_tracker_kmeans.txt"
    myFile5 = "mass_shootings_dbscan.txt"
    myFile6 = "mass_shooting_tracker_dbscan.txt"

    ###########
    # READ DATA
    ###########

    # Dataframe for main shooting data
    # Thows FileNotFoundError in VS Code, but works in Spyder
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

    # Categorize numerical data
    df1 = categorize1(df1)
    df2 = categorize2(df2)

    # Drop certain columns
    df1 = choose_features1(df1)
    df2 = choose_features2(df2)

    # Normalize data
    x_scaled1, normalizedDataFrame1 = preprocess(df1)
    x_scaled2, normalizedDataFrame2 = preprocess(df2)

    ######
    # WARD
    ######

    agglo(x_scaled1, normalizedDataFrame1, 3, myFile1, True)
    agglo(x_scaled1, normalizedDataFrame1, 7, myFile1, True)
    agglo(x_scaled1, normalizedDataFrame1, 12, myFile1, True)

    agglo(x_scaled2, normalizedDataFrame2, 5, myFile2, False)
    agglo(x_scaled2, normalizedDataFrame2, 10, myFile2, False)
    agglo(x_scaled2, normalizedDataFrame2, 15, myFile2, False)
    agglo(x_scaled2, normalizedDataFrame2, 20, myFile2, False)

    ########
    # KMEANS
    ########

    kmeans(x_scaled1, normalizedDataFrame1, 3, myFile3, True)
    kmeans(x_scaled1, normalizedDataFrame1, 7, myFile3, True)
    kmeans(x_scaled1, normalizedDataFrame1, 12, myFile3, True)

    kmeans(x_scaled2, normalizedDataFrame2, 5, myFile4, False)
    kmeans(x_scaled2, normalizedDataFrame2, 10, myFile4, False)
    kmeans(x_scaled2, normalizedDataFrame2, 15, myFile4, False)
    kmeans(x_scaled2, normalizedDataFrame2, 20, myFile4, False)

    ########
    # DBSCAN
    ########

    dbscan(x_scaled1, normalizedDataFrame1, 30, myFile5, True)
    dbscan(x_scaled1, normalizedDataFrame1, 16, myFile5, True)
    dbscan(x_scaled1, normalizedDataFrame1, 9, myFile5, True)

    dbscan(x_scaled2, normalizedDataFrame2, 1000, myFile6, False)
    dbscan(x_scaled2, normalizedDataFrame2, 500, myFile6, False)
    dbscan(x_scaled2, normalizedDataFrame2, 250, myFile6, False)
    dbscan(x_scaled2, normalizedDataFrame2, 100, myFile6, False)


if __name__ == "__main__":
    main()
