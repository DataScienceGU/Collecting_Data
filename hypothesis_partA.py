import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing


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
    names = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-100']
    bins = [0, 18, 25, 35, 45, 55, 100]
    myDataFrame['age_of_shooter_binned'] = pd.cut(
        myDataFrame['age_of_shooter'], bins, labels=names)
    return myDataFrame


def binTotalVictims(myDataFrame):
    names = ['0-6', '6-15', '>15']
    bins = [0, 6, 15, 10000]
    # bins the total victims into 3 equal width bins
    myDataFrame['total_victims_binned'] = pd.cut(
        myDataFrame['total_victims'], bins=bins, labels=names)
    return myDataFrame


def choose_features1(df):
    # Select the columns/features to use for training data
    df = pd.concat([df['injured'], df['fatalities'], df['date'], df['age_of_shooter_binned']],
                   axis=1)

    df = extract_year(df)

    return df


def choose_features2(df):
    # Select the columns/features to use
    df = pd.concat([df['wounded'], df['killed'], df['date']],
                   axis=1)

    df.rename(columns={'wounded': 'injured',
                       'killed': 'fatalities'}, inplace=True)

    df = extract_year(df)

    df["age_of_shooter_binned"] = ""

    return df


def extract_year(df):
    # Converts non-numerical data to numerical, using pandas

    # Extract just the year from the date
    df['date'] = pd.DatetimeIndex(df['date']).year

    return df


def main():
    training_data = pd.read_csv("mass_shootings.csv")
    test_data = pd.read_csv('mass_shooting_tracker_2013.csv',
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
    test_data = test_data.append(temp1)
    test_data = test_data.append(temp2)
    test_data = test_data.append(temp3)
    test_data = test_data.append(temp4)
    test_data = test_data.append(temp5)
    test_data = test_data.append(temp6)

    training_data = getValsForGunDescription(training_data)
    training_data = changeMentalHealthVals(training_data)
    training_data = binAges(training_data)
    training_data = binTotalVictims(training_data)

    training_data = choose_features1(training_data)
    test_data = choose_features2(test_data)

    # print(training_data[:20])
    # print(test_data)

    ######################################################
    # Evaluate algorithms
    ######################################################

    # Separate training and final validation data set. First remove class
    # label from data (X). Setup target class (Y)
    # Then make the validation set 20% of the entire
    # set of labeled data (X_validate, Y_validate)
    training_value_array = training_data.values
    test_value_array = test_data.values

    X_train = training_value_array[:, 0:-1]
    preprocessing.normalize(X_train)
    Y_train = training_value_array[:, -1]
    X_validate = test_value_array[:, 0:-1]
    Y_validate = test_value_array[:, -1]

    test_size = 0.20
    seed = 7

    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'

    ######################################################
    # Use different algorithms to build models
    ######################################################

    # Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ADA', AdaBoostClassifier()))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # ######################################################
    # # For the best model (KNN), see how well it does on the
    # # validation test
    # ######################################################
    # # Make predictions on validation dataset
    # knn = KNeighborsClassifier()
    # knn.fit(X_train, Y_train)
    # predictions = knn.predict(X_validate)

    # print()
    # print('Accuracy score for KNN')
    # print("{0:.6f}".format(accuracy_score(Y_validate, predictions)))
    # print(confusion_matrix(Y_validate, predictions))
    # print(classification_report(Y_validate, predictions))

    # ######################################################
    # # For the model (Decision Tree), see how well it does on the
    # # validation test
    # ######################################################
    # # Make predictions on validation dataset
    # dectree = DecisionTreeClassifier()
    # dectree.fit(X_train, Y_train)
    # predictions = dectree.predict(X_validate)

    # print()
    # print('Accuracy score for tree')
    # print("{0:.6f}".format(accuracy_score(Y_validate, predictions)))
    # print(confusion_matrix(Y_validate, predictions))
    # print(classification_report(Y_validate, predictions))
    # # Decision tree
    # # http://scikit-learn.org/stable/modules/tree.html
    # # A Lazy Learner Method (such as kNN)
    # #  http://scikit-learn.org/stable/modules/neighbors.html
    # # http://scikitlearn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
    # #  Naïve Bayes
    # # http://scikit-learn.org/stable/modules/naive_bayes.html
    # # Random Forest
    # # http://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # # ** add on one extra model**
if __name__ == "__main__":
    main()
