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


def choose_features1(df, classifier):
    # Select the columns/features to use for training data
    df = pd.concat([df['injured'], df['fatalities'], df['date'], df[classifier]],
                   axis=1)

    df = extract_year(df)

    return df


def choose_features2(df, classifier):
    # Select the columns/features to use
    df = pd.concat([df['wounded'], df['killed'], df['date']],
                   axis=1)

    df.rename(columns={'wounded': 'injured',
                       'killed': 'fatalities'}, inplace=True)

    df = extract_year(df)

    df[classifier] = ""

    return df


def extract_year(df):
    # Converts non-numerical data to numerical, using pandas

    # Extract just the year from the date
    df['date'] = pd.DatetimeIndex(df['date']).year

    return df


def main():
    # Filenames
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

    save_name = 'mass_shooting_predictions.csv'

    # Combine dataframes for extra shooting data
    test_data = test_data.append(temp1)
    test_data = test_data.append(temp2)
    test_data = test_data.append(temp3)
    test_data = test_data.append(temp4)
    test_data = test_data.append(temp5)
    test_data = test_data.append(temp6)

    test_data_temp = test_data

    training_data = getValsForGunDescription(training_data)
    training_data = changeMentalHealthVals(training_data)
    training_data = binAges(training_data)

    # README: Here we can choose which classifier to predict, uncomment/comment the lines that we
    # want to predict

    # training_data = choose_features1(training_data, 'age_of_shooter_binned')
    # test_data_temp = choose_features2(test_data_temp, 'age_of_shooter_binned')

    training_data = choose_features1(
        training_data, 'auto_or_semiauto_or_rifle')
    print(training_data)
    test_data_temp = choose_features2(test_data, 'auto_or_semiauto_or_rifle')
    print(test_data_temp)

    # training_data = choose_features1(
    #     training_data, 'has_mental_health_issues')
    # test_data_temp = choose_features2(
    #     test_data, 'has_mental_health_issues')

    ######################################################
    # Evaluate algorithms
    ######################################################

    training_value_array = training_data.values
    test_value_array = test_data_temp.values

    X_train = training_value_array[:, 0:-1]
    preprocessing.normalize(X_train)
    Y_train = training_value_array[:, -1]
    X_validate = test_value_array[:, 0:-1]
    Y_validate = test_value_array[:, -1]

    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
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
    models.append(('RF', RandomForestClassifier(n_estimators=100)))
    models.append(('ADA', AdaBoostClassifier()))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    seed = 7
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    #################################################################
    # Use the best model (Random Forest), to make predictions on 3
    # different classifiers:
    #################################################################

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_validate)

    # Comment/uncomment the classifier we want to test
    # test_data['age_of_shooter_binned'] = predictions
    test_data['auto_or_semiauto_or_rifle'] = predictions
    # test_data['has_mental_health_issues'] = predictions

    test_data.to_csv(save_name, encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
