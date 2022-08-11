#!/usr/bin/env python

'''
    Photometric Approximator of Stellar Metallicity (c) by Rik Ghosh, Soham Saha

    Photometric Approximator of Stellar Metallicity is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this work.
    If not, see https://creativecommons.org/licenses/by/4.0
'''

'''
    This python script is used to preprocess the training (and evaluation) data so
    that extreme outliers can be removed without losing any valuable information

    This file should not need any further editing from the side of the user
'''

# imports
import pandas as pd
import numpy as np

# custom library imports
from visualize import boxplot


# preprocess color data to remove extreme outliers through repetitive median reductions
def remove_outliers(df: pd.DataFrame, args: dict = {} ) -> pd.DataFrame:
    '''
    Remove extreme outliers per repetition of the input data

    Args:
        df (pandas.DataFrame): input dataframe
        args (dict): customizable method arguments passed in as a dictionary

    Returns:
        pd.DataFrame: dataframe with outliers removed
    '''

    # base arguments dictionary -> modified by args parameter
    _args = {
        'iqr_factor' : 1.5,             # only (-iqr_factor * IQR + Q1, iqr_factor * IQR + Q3) retained
        'num_reps'   : 1,               # number of times to repeat the outlier exclusion
        'show'       : False,           # boolean to display box and whisker plots
        'save'       : False,           # boolean to save figure
        'dpi'        : 300              # save quality
    }

    for key in args.keys():
        _args[key] = args[key]      # update parameters

    COLORS = ['ug', 'gr', 'ri', 'iz']   # expected colors in the input dataframe

    try:
        color_data = [df[x] for x in COLORS]
    except KeyError:
        print('The input dataframe is missing the one of the following columns:')
        print([x for x in COLORS])

    # repeat eliminations
    while _args['num_reps']:
        mins = list()
        maxs = list()

        for elem in COLORS:
            q3 = np.percentile(df[elem], 75)
            q1 = np.percentile(df[elem], 25)
            IQR = q3 - q1

            mins.append(q1 - _args['iqr_factor'] * IQR)
            maxs.append(q3 + _args['iqr_factor'] * IQR)
        
        min_est = min(mins)
        max_est = max(maxs)

        # throw away outliers
        for elem in COLORS:
            df = df[(df[elem] >= min_est) & (df[elem] <= max_est)]

        _args['num_reps'] -= 1

    processed_data = [df[x] for x in COLORS]

    if _args['show']:
        boxplot(color_data, processed_data, _args)

    return df
