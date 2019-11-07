import pandas as pd
from statistics import mode, median
import numpy as np

moviedata = pd.read_csv("all_movies.csv")
#ms = mass shooting
msdata = pd.read_csv("mass_shootings.csv")

#med = median
#mean or mode
#standard deviation

#va = vote average
va_mean = moviedata['vote_average'].mean()
va_med = moviedata['vote_average'].median()
va_std = moviedata['vote_average'].std()
print("vote average mean is %s" % va_mean)
print("vote average median is %s" % va_med)
print("vote average standard deviation is %s" % va_std)
print()

#vc = vote count
vc_mean = moviedata['vote_count'].mean()
vc_med = moviedata['vote_count'].median()
vc_std = moviedata['vote_count'].std()
print("vote count mean is %s" % vc_mean)
print("vote count median is %s" % vc_med)
print("vote count standard deviation is %s" % vc_std)
print()

#pop = popularity
pop_mean = moviedata['popularity'].mean()
pop_med = moviedata['popularity'].median()
pop_std = moviedata['popularity'].std()
print("popularity mean is %s" % pop_mean)
print("popularity median is %s" % pop_med)
print("popularity standard deviation is %s" % pop_std)
print()

#ft = fatalities
ft_mean = msdata['fatalities'].mean()
ft_med = msdata['fatalities'].median()
ft_std = msdata['fatalities'].std()
print("fatalities mean is %s" % ft_mean)
print("fatalities median is %s" % ft_med)
print("fatalities standard deviation is %s" % ft_std)
print()

#in = injured
in_mean = msdata['injured'].mean()
in_med = msdata['injured'].median()
in_std = msdata['injured'].std()
print("injured mean is %s" % in_mean)
print("injured median is %s" % in_med)
print("injured standard deviation is %s" % in_std)
print()

#tv = total injured
tv_mean = msdata['total_victims'].mean()
tv_med = msdata['total_victims'].median()
tv_std = msdata['total_victims'].std()
print("total victims mean is %s" % tv_mean)
print("total victims median is %s" % tv_med)
print("total victims standard deviation is %s" % tv_std)
print()

#aos = age of shooter
aos_mean = msdata['age_of_shooter'].mean()
aos_med = msdata['age_of_shooter'].median()
aos_std = msdata['age_of_shooter'].std()
print("age of shooter mean is %s" % aos_mean)
print("age of shooter median is %s" % aos_med)
print("age of shooter standard deviation is %s" % aos_std)
print()

#y = year
y_mean = msdata['year'].mean()
y_med = msdata['year'].median()
y_std = msdata['year'].std()
print("year of shooting mean is %s" % y_mean)
print("year of shooting median is %s" % y_med)
print("year of shooting standard deviation is %s" % y_std)
print()

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
moviedata['pop_mtm'] = np.where(moviedata['popularity']>=pop_med, 'yes', 'no')
print(moviedata['pop_mtm'])






