import glob
import pandas as pd
import pylab as pl
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def load_gun_sheets():
    path = 'GunViolenceData'
    filenames = glob.glob(path+"/*.csv")
    filenames.sort()

    print(filenames)

    dataframes = []

    for filename in filenames:
        df = pd.read_csv(filename)
        dataframes.append(df)

    all_shooting_data = pd.concat(dataframes, ignore_index=True)
    all_shooting_data.head()

    return all_shooting_data

def plot_histograms(shooting_data):
    #using date as last histogram value from this database, separating month over month per year
    #i figured day by day was too granular.
    shooting_data['date'] = shooting_data['date'].astype('str')
    print(shooting_data['date'].dtype)
    month_year = []
    for date in shooting_data['date']:
        datelist = date.split('/')
        month_year.append((datelist[0] + '/' + datelist[2]))

    print(month_year)
    shooting_data['month_year'] = month_year
    # shooting_data['month_year'] = shooting_data['month_year'].astype('category').cat.codes

    print(shooting_data)
    shooting_data.hist()
    plt.show()
    plt.clf()
    plt.close()

    name = "killed"
    shooting_data["killed"].hist()
    pl.suptitle("# Killed per Shooting")
    plt.savefig(name)
    plt.clf()
    plt.close()

    name = "wounded"
    shooting_data["wounded"].hist()
    pl.suptitle("# Wounded per Shooting")
    plt.savefig(name)
    plt.clf()
    plt.close()

    name = "frequency"
    shooting_data["month_year"].hist()
    pl.suptitle("# of Shootings per Month")
    plt.savefig(name)
    plt.clf()
    plt.close()


def main():
    shooting_data = load_gun_sheets()
    print(shooting_data)
    print("killed max: ")
    print(shooting_data['killed'].max())
    print("wounded max: ")

    print(shooting_data['wounded'].max())
    plot_histograms(shooting_data)



if __name__ == "__main__":
    main()