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


def getValsForGunDescription(myDataFrame):
    # check substring of gun description
    # return new column that is yes or no
    myDataFrame.loc[myDataFrame['weapon_type'].str.contains(
        pat="auto"), 'auto_or_semiauto_or_rifle'] = 'Strong_Gun'
    myDataFrame.loc[myDataFrame['weapon_type'].str.contains(
        pat="rifle"), 'auto_or_semiauto_or_rifle'] = 'Strong_Gun'
    myDataFrame.loc[myDataFrame['weapon_type'].str.contains(
        pat="shotgun"), 'auto_or_semiauto_or_rifle'] = 'Strong_Gun'
    myDataFrame['auto_or_semiauto_or_rifle'] = myDataFrame['auto_or_semiauto_or_rifle'].fillna(
        "Weak_Gun")

    return myDataFrame


def changeMentalHealthVals(myDataFrame):
    # have mental health issues be a yes or no
    myDataFrame.loc[myDataFrame['prior_signs_mental_health_issues']
                    == 'TBD', 'has_mental_health_issues'] = 'No_Mental_Issues'
    myDataFrame.loc[myDataFrame['prior_signs_mental_health_issues'] ==
                    'Unclear', 'has_mental_health_issues'] = 'No_Mental_Issues'
    myDataFrame.loc[myDataFrame['prior_signs_mental_health_issues'] ==
                    'Yes', 'has_mental_health_issues'] = 'Yes_Mental_Issues'
    myDataFrame['has_mental_health_issues'] = myDataFrame['has_mental_health_issues'].fillna(
        "No_Mental_Issues")
    return myDataFrame


def binAges(myDataFrame):
    # bins the ages into separate groups
    # names = range(0, 6)  # names list from 0 to 5
    names = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-100']
    bins1 = [0, 18, 25, 35, 45, 55, 100]
    myDataFrame['age_of_shooter_binned'] = pd.cut(
        myDataFrame['age_of_shooter'], bins1, labels=names)
    return myDataFrame


def binTotalVictims(myDataFrame):
    names = ['0-6', '6-15', '>15']
    bins = [0, 6, 15, 10000]
    # bins the total victims into 3 equal width bins
    myDataFrame['total_victims_binned'] = pd.cut(
        myDataFrame['total_victims'], bins=bins, labels=names)
    return myDataFrame


def grabColumnsOfInterest(myDataFrame):
    # returns a dataframe with only the columns that will be looked at for rule mining
    return myDataFrame[['race',  'auto_or_semiauto_or_rifle', 'has_mental_health_issues', 'age_of_shooter_binned', 'total_victims_binned']]


def prepareForApriori():
    # will run specific functions to prepare the data to run the apriori
    # function on it
    filename = "mass_shootings.csv"
    myDataFrame = readData(filename)
    myDataFrame = changeMentalHealthVals(myDataFrame)
    myDataFrame = getValsForGunDescription(myDataFrame)
    myDataFrame = binAges(myDataFrame)
    myDataFrame = binTotalVictims(myDataFrame)
    myDataFrame = myDataFrame.dropna()
    myDataFrame = grabColumnsOfInterest(myDataFrame)
    print(myDataFrame)
    return myDataFrame


def runAprioriAlg(myDataFrame, minSupport):
    # runs the apriori algorithm based on the min_support passed in and
    # returns the result

    # have min_lift equal to a number barely over 1 so that it looks at
    # only itemsets with more than 1 item
    results = list(apriori(myDataFrame.values,
                           min_support=minSupport, min_lift=2.000001))

    print(results)


def main():
    myDataFrame = prepareForApriori()
    runAprioriAlg(myDataFrame, 0.08)


if __name__ == "__main__":
    main()
