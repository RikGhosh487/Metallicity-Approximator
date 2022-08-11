#!/usr/bin/env python

'''
    Photometric Approximator of Stellar Metallicity (c) by Rik Ghosh, Soham Saha

    Photometric Approximator of Stellar Metallicity is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this work.
    If not, see https://creativecommons.org/licenses/by/4.0
'''

'''
    Use this python script to randomly shuffle the contents of the train and the valid datasets.
    
    If you wish to keep the sizes of each of the datasets the same, do not change the ratios used
    in this file.
'''

# third party imports
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# META DATA
FILE_PATH = './segue.csv'
Y_COL_NAME = 'feh'
TEST_RATIO = 0.2

if __name__ == '__main__':
    df = read_csv(FILE_PATH)
    X = df.drop(Y_COL_NAME, axis=1)         # separate x and y values
    Y = df[Y_COL_NAME]

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=TEST_RATIO)

    # recombine with appropriate pairs to produce final datasets
    x1[Y_COL_NAME] = y1
    x2[Y_COL_NAME] = y2

    x1.to_csv('./train/data.csv', index=False)
    x2.to_csv('./valid/data.csv', index=False)
