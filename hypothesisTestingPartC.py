import pandas as pd
import numpy as np
import pprint
import copy
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


def readData(filename):
    # read the csv data and return a pandas dataframe
    myDataFrame = pd.read_csv(
        filename, sep=",", encoding='latin1')
    return myDataFrame


def getListOfViolentGenres():
    # Types of movies that are violent:
    # these types of movies tend to include more violence than other genres
    # Action(28), adventure(12), horror(27), thriller(53), western(37), crime (80), war(10752)
    return [28, 12, 27, 53, 37, 878, 80, 10752]
    # return [27, 53]
    # return [28]


def getListOfViolentWords():
    # possible violent words
    violent_words = ["gun", "shoot", "murder", "war", "kill", "pistol", "massacre",
                     "rampage", "violent", "hunt", "mafia", "attack", "police", "assassin", "crime"]
    # violent_words = ["gun"]
    return violent_words


def findViolentProportionByMonth(myDataFrame):
    # return a dict that shows the proportion of violent movies according to year
    # grab unique year values as keys for the dict
    yearList = myDataFrame.release_year.unique()
    # dict that has year as key and proportion as value
    proportionDict = {}
    # now, loop through yearList and the number of violent movies from that year over all movies of that year
    for year in yearList:
        allCount = 0
        violentCount = 0
        for index, row in myDataFrame.iterrows():
            if row['release_year'] == year:
                # increment allCount, increment violentCount if violent is 1
                allCount += 1
                if row['violent_1_or_0'] == 1:
                    violentCount += 1
        # now find the proportion
        proportionOfViolentInYear = violentCount/allCount
        proportionDict[year] = proportionOfViolentInYear
    return proportionDict


def returnSortedListOfYears(proportionDict):
    # returns the list of years in numerical order
    return sorted(proportionDict)


def returnSortedListOfProportions(proportionDict):
    # returns the list of proportions in the same order as the years
    proportionList = []
    for x in sorted(proportionDict):
        proportionList.append(proportionDict[x])
    return proportionList


def createMonthYearColumn(myDataFrame):
    # creates a new column that is purely the the year of release
    myDataFrame['release_date'] = myDataFrame['release_date'].astype('str')
    yearList = []
    for date in myDataFrame['release_date']:
        datelist = date.split('-')
        yearList.append(datelist[0])
    myDataFrame['release_year'] = yearList
    return myDataFrame


def convertStringIntoListForGenres(myDataFrame):
    # converts a string represenation of a list into an actual list
    myDataFrame['genre_ids'] = myDataFrame['genre_ids'].apply(
        lambda x: x.strip('][').split(', '))
    return myDataFrame


def createBinaryViolentMovieColumnByGenre(myDataFrame, listOfViolentGenres):
    # will create a new column that is 1 if movie has a violent genre and 0 otherwise
    myDataFrame['violent_1_or_0'] = myDataFrame['genre_ids'].apply(
        lambda x: 1 if doSomething(x, listOfViolentGenres) else 0)
    return myDataFrame


def doSomething(List1, List2):
    # Found here: https://www.techbeamers.com/program-python-list-contains-elements/#any-method
    # Checks to see if any part of list 2 is contained in list 1
    # returns true if there is a match, false otherwise
    match = False
    for x in List1:
        for y in List2:
            # needed to convert to a string because otherwise equality method did not work
            if str(x) == str(y):
                match = True
                return match

    return match


def returnTwoSetsOfFloatDataByDict(proportionDict):
    # takes in a dict, sorts by key, returns two lists (keys and values),
    # and converts them to floats for correlation test
    sortedListOfYears = returnSortedListOfYears(copy.copy(proportionDict))
    sortedListOfProportions = returnSortedListOfProportions(
        copy.copy(proportionDict))
    sortedListOfYears = list(map(float, sortedListOfYears))
    sortedListOfProportions = list(map(float, sortedListOfProportions))
    return sortedListOfYears, sortedListOfProportions


def createBinaryViolentMovieColumnByOverview(myDataFrame, listOfViolentWords):
    # will hold truth value if violent or not.
    contains_violence = []

    for movie in myDataFrame['overview']:
        violent = 0
        for word in listOfViolentWords:
            if word in movie:
                violent = 1

        contains_violence.append(violent)

    # add the column into the dataframe
    myDataFrame['violent_1_or_0'] = contains_violence
    return myDataFrame


def runCorrelationOnData(myDataFrame):
    # will prepare data for a t-test / correlation
    # the listToCheck is the list that will define the binary violent column
    findViolentProportionByMonth(myDataFrame)
    proportionDict = findViolentProportionByMonth(myDataFrame)
    sortedListOfYears, sortedListOfProportions = returnTwoSetsOfFloatDataByDict(
        proportionDict)
    corr, p = pearsonr(sortedListOfYears, sortedListOfProportions)
    return corr, p


def prepareDataFrame():
    # will read the data and prepare the data for running correlation tests
    myDataFrame = readData('all_movies.csv')
    myDataFrame = convertStringIntoListForGenres(myDataFrame)
    myDataFrame = createMonthYearColumn(myDataFrame)
    return myDataFrame


def main():
    # runs a linear correlation test for the proportion of violent movies over
    # time by genre and then by looking at violent words in the overview description
    myDataFrame = prepareDataFrame()
    listOfViolentGenres = getListOfViolentGenres()
    myDataFrameGenre = createBinaryViolentMovieColumnByGenre(
        myDataFrame, listOfViolentGenres)
    corr, p = runCorrelationOnData(myDataFrameGenre)
    print(corr)
    print(p)
    listOfViolentWords = getListOfViolentWords()
    myDataFrameOverview = createBinaryViolentMovieColumnByOverview(
        myDataFrame, listOfViolentWords)
    corr, p = runCorrelationOnData(myDataFrameOverview)
    print(corr)
    print(p)


if __name__ == "__main__":
    main()
