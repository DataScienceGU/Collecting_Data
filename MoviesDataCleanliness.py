import pandas as pd

# Specifically, you should identify missing and incorrect values.
# You can then record:
# • The fraction of missing values for each attribute.
# • The fraction of noise values, e.g. gender = ‘fruit’.

missing_attributes = {
    'Unnamed: 0' : 0, 'adult' : 0, 'backdrop_path' : 72, 'genre_ids' : 0, 'id' : 0, 'original_language' : 0,
    'original_title' : 0, 'overview' : 26, 'popularity' : 0, 'poster_path' : 9, 'release_date' : 0, 'title' : 0,
    'video' : 0, 'vote_average' : 0, 'vote_count' : 0}

def main():
    # Read in data as a pandas dataframe
    totalDataFrameLength = 0
    for i in range(1,5):
        #if the first sheet, don't append the number (because we didn't do that manually)
        if i > 1:
            filename = "movies" + str(i) + ".csv"
        else:
            filename = "movies.csv"

        #load csv to dataframe, append record length to running total of records across the movie csvs
        df = pd.read_csv(filename, sep=',', encoding='latin1')
        totalDataFrameLength +=len(df)

        #loading the sum of null values in sheet as a series
        nullSeries = df.isnull().sum()

        #adding to running totals in dictionary.
        for attr in nullSeries.index:
            missing_attributes[attr] += nullSeries[attr]

    # Write summary of bad data to file
    with open("badData.txt", 'a') as f:
        f.write('\n=========================================================================================\n')
        f.write('Bad data summary for movie data: \n')
        for attr in missing_attributes:
            f.write('Missing values for ' + attr +
                    ': ' + str(missing_attributes[attr]) + '\n')

        f.write('\nFraction of missing values for each attribute:\n')
        for attr in missing_attributes:
            f.write(attr + ': ' +
                    str(missing_attributes[attr] / totalDataFrameLength) + '\n')

        f.write('\nThere were no noise values to measure in this data.')

if __name__ == '__main__':
    main()