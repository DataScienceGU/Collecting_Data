import pandas as pd

in_filename = 'mass_shootings.csv'
bad_data_filename = 'badData.txt'

# Specifically, you should identify missing and incorrect values.
# You can then record:
# • The fraction of missing values for each attribute.
# • The fraction of noise values, e.g. gender = ‘fruit’.

# Missing values are recorded as the following in the data
missing_vals = ['-', 'TBD', 'Unclear', 'Unknown']

# The following attributes have missing values in the data
missing_attributes = {'prior_signs_mental_health_issues': 0, 'mental_health_details': 0,
                      'weapons_obtained_legally': 0, 'where_obtained': 0, 'weapon_details': 0, 'race': 0,
                      'mental_health_sources': 0, 'sources_additional_age': 0}


def main():
    # Read in data as a pandas dataframe
    df = pd.read_csv(in_filename, sep=',', encoding='latin1')
    
    # If a missing attribute is found, increase its counter in the dictionary
    for index, row in df.iterrows():
        for attr in missing_attributes:
            for val in missing_vals:
                if (row[attr] == val):
                    missing_attributes[attr] += 1

    # Write summary of bad data to file
    with open(bad_data_filename, 'w') as f:
        for attr in missing_attributes:
            f.write('Missing values for ' + attr +
                    ': ' + str(missing_attributes[attr]) + '\n')

        f.write('\nFraction of missing values for each attribute:\n')
        for attr in missing_attributes:
            f.write(attr + ': ' +
                    str(missing_attributes[attr]/len(df)) + '\n')

        f.write('\nThere were no noise values to measure in this data.')


if __name__ == '__main__':
    main()
