import pandas as pd
from statistics import mode, median
import numpy as np

def getStats(df, var):
    mean = df[var].mean()
    median = df[var].median()
    std = df[var].std()
    print(var + " stats:\n")
    print("mean is %s" % mean)
    print("median is %s" % median)
    print("standard deviation is %s" % std)
    print()


def main():
    moviedata = pd.read_csv("all_movies.csv")
    #ms = mass shooting
    msdata = pd.read_csv("mass_shootings.csv")

    #mean or mode
    #median
    #standard deviation

    getStats(moviedata, 'vote_average')
    getStats(moviedata, 'vote_count')
    getStats(moviedata, 'popularity')
    getStats(msdata, 'fatalities')
    getStats(msdata, 'injured')
    getStats(msdata, 'total_victims')
    getStats(msdata, 'age_of_shooter')
    getStats(msdata, 'year')

    #r = race of shooter
    r_mode = mode(msdata['race'])
    r_med = median(msdata['race'])
    #no std, not a numeric value
    print("race of shooter mode is %s" % r_mode)
    print("race of shooter median is %s" % r_med)
    print("no standard deviation, not a numeric value")
    print()

    #wt = weapon type
    wt_mode = mode(msdata['weapon_type'])
    wt_med = median(msdata['weapon_type'])
    #no std, not a numeric value
    print("weapon type mode is %s" % wt_mode)
    print("weapon type median is %s" % wt_med)
    print("no standard deviation, not a numeric value")
    print()


    #############################
    # check if row value has more popularity than median
    # mtm = more than median
    pop_med = moviedata['popularity'].median()
    moviedata['pop_mtm'] = np.where(moviedata['popularity']>=pop_med, 'yes', 'no')
    

if __name__ == "__main__":
    main()





