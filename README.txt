Boya Lee, Michael Castano, Karina Chan, Philipp Seitz

Source Files: 

“basic_stats.py” - returns the mean/mode, median, standard deviation (if any) for 10 different numerical 
and categorical attributes across both movie and mass shooting datasets as well as a new binary column 
based on if popularity attribute in movie dataset is above median or not

cluster_analysis_shootings.py - performs cluster analysis on shooting data, also plots 2D PCA graphs for 
each algorithm/dataset used. Outputs using the original data set will have “original” in file names, while 
outputs dealing with extra shooting data will have “extra” in file names. Comment out the runB to get 
textfiles for clustering outputs for the original data.

cluster_analysis_movies.py - performs cluster analysis on movie data, also plots 2D PCA graphs for each algorithm used

mass_shootings_ALGORITHM.txt
mass_shooting_tracker_ALGORITHM.txt
movie_ALGORITHM.txt
- are all written results of clustering algorithms.

Figures: 
	Clustering:
		shootings_pca_original_ALGORITHM_#.png
		shootings_pca_extra_ALGORITHM_#.png
		movie_pca_ALGORITHM_#.png
			- 2D PCA graphs of all the different combinations of clustering algorithms 
              and data sets

    Histograms: 
        frequency_shootings.png - histogram of frequency of shootings per month from 2013 to 2019. 
        killed.png - histogram of a number of shootings with a certain number of people killed per shooting
        wounded.png - histogram of a number of shootings with a certain number of people wounded per shooting 
        violent_movies.png - histogram of the number of violent and non violent movies in the data set
        violent_words.png - histogram of the number of movies with the number of violent words per description
        genres.png - histogram with the number of movies released by genre. 
    Scatterplots: 
        correlation_movie_genre_shootings.png - scatterplot of the number of movies in sensational genres released the same month as shootings. 
        correlation_movie_genre_violent_movie.png - scatterplot of the number of movies in sensational genres released the same month as violent movies. 
        correlation_movies_shootings.png - scatterplot of the number of violent movies genres released the same month as shootings. 
	Linear Regression: 
        linear_regression_violent_movies_shootings.png - linear regression of the plot of correlation_movies_shootings.png 

ruleMiningForMovies.py: returns the results of running an apriori algorithm on the all_movies.csv dataset. 
The given columns looked at are explained in the write-up. Simply change the values for min_support and/or 
min_lift to print out the desired results

ruleMiningForShootings.py:  returns the results of running an apriori algorithm on the mass_shootings.csv 
dataset. The given columns looked at are explained in the write-up. Simply change the values for 
min_support and/or min_lift to print out the desired results

hypothesis_partA.py: Makes predictions for 3 classifiers for the extra shooting data that was used as 
test data, with the original shooting data used as training data. There are parts indicated in the code 
where you can comment/uncomment to test different classifications.

Histograms_correlations_hypothesis_b.py - creates 6 histograms, 3 scatter plots, and one linear regression to test 
our second hypothesis. I did this because I conducted a linear regression on two variables that were used in computing 
one of my correlations so I didn’t want to create a whole new file and rebuild and reformat another data frame instead of 
doing it in one file. 

hypothesisTestingPartC: return the correlation factor and p-value for hypothesis testing according to the proportion of violent movies by time. Does this twice where it categorizes violent movies by genres and another one by keyword in the movie overview. 
