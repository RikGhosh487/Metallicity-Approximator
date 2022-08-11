#!/usr/bin/env python

'''
    Photometric Approximator of Stellar Metallicity (c) by Rik Ghosh, Soham Saha

    Photometric Approximator of Stellar Metallicity is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this work.
    If not, see https://creativecommons.org/licenses/by/4.0
'''

'''
    This python script contains the main components of the training and evaluation logic.
    In order to use the model in the future, please refer the the model_predict() function

    This file should not need any further editing from the side of the user
'''

# standard library imports
import os

# imports
import sklearn.ensemble as en
from copy import deepcopy
import typing as tp
import pandas as pd
import numpy as np
import pickle

# custom library imports
from preprocess import remove_outliers
from visualize import truth_pred_scat, truth_pred_side

# META DATA
default_savename = 'model.sav'

# saving utility function
def save_model(curr_dir: str, model: en.RandomForestRegressor) -> None:
    '''
    Saves the model so that it can be loaded in when required without needing training again
    
    Args:
        curr_dir (str)                  : name of the current directory
        model (en.RandomForestRegressor): the model to save
        
    Returns:
        None
    '''

    save_dir = os.path.join(curr_dir, 'models')             # directory for models

    if not os.path.exists(save_dir):                        # check to see if the directory already exists
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, default_savename)
    pickle.dump(model, open(save_path, 'wb'))

    print('Saved model in models/%s' % default_savename)


# loading utility function
def load_model(curr_dir: str, savename: str) -> en.RandomForestRegressor:
    '''
    Loads a saved model so that it can be used
    
    Args:
        curr_dir (str): name of the current directory
        savename (str): name of the saved model file
    
    Returns:
        sklearn.ensemble.RandomForestRegressor: the saved model
    '''
    
    save_dir = os.path.join(curr_dir, 'models')             # directory for models
    if not os.path.exists(save_dir):
        raise FileNotFoundError('There is no such directory called ./models')

    save_path = os.path.join(save_dir, savename)
    model = pickle.load(open(save_path, 'rb'))

    return model


# main training function
def train(data_dir: str, args: dict = {}) -> en.RandomForestRegressor:
    '''
    Creates and trains the random forest regressor model using data from the specified path
    
    Args:
        data_dir (str): path to the data source to train on
        args (dict)   : customizable method arguments passed in as a dictionary
    
    Returns:
        sklearn.ensemble.RandomForestRegressor: the trained model
    '''

    _args = {
        'n_estimators': 10,
        'random_state': 0,
        'iqr_factor'  : 1.5,
        'filename'    : '',
        'num_reps'    : 1,
        'show'        : False,
        'colors'      : [],
        'save'        : False,
        'save_model'  : False,
        'dpi'         : 300
    }

    for key in args.keys():
        _args[key] = args[key]      # update parameters

    if _args['filename'] == '':
        # try default filename
        _args['filename'] = 'data.csv'

    # find data directory
    curr_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    target_file = os.path.join(curr_dir, data_dir, 'train', _args['filename'])

    if not os.path.exists(target_file):
        raise FileNotFoundError('The file specified does not exist')
    
    df = pd.read_csv(target_file)

    # pre-processing
    df = remove_outliers(df, _args)

    # split parameters from truth values
    y = pd.DataFrame(df.pop('feh'))
    x = df

    # model creation
    model = en.RandomForestRegressor(n_estimators=_args['n_estimators'], random_state=_args['random_state'])
    model.fit(x, y.values.ravel())

    if _args['save_model']:
        save_model(curr_dir, model)

    return model


# scoring and evaluating function
def score_eval(model: tp.Union[en.RandomForestRegressor, str], args: dict = {}) -> None:
    '''
    Scores the accuracy of the trained model and provides evaluation metrics for the tested data

    Args:
        model (en.RandomForestRegressor | str) : the model to evaluate itself, or 'load' indicating
                                                 the model needs to be loaded in
        args (dict)                            : customizable method arguments passed in as a dictionary

    Returns:
        None
    '''

    _args = {
        'iqr_factor'  : 1.5,
        'load_name'   : '',
        'data_dir'    : '',
        'filename'    : '',
        'num_reps'    : 1,
        'show'        : False,
        'colors'      : [],
        'save'        : False,
        'save_model'  : False,
        'dpi'         : 300
    }

    for key in args.keys():
        _args[key] = args[key]      # update parameters

    if _args['filename'] == '':
        # try default filename
        _args['filename'] = 'data.csv'

    if _args['load_name'] == '':
        # try default load_name
        _args['load_name'] = default_savename

    curr_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

    if isinstance(model, str):
        if model != 'load':
            raise ValueError('unknown command: %s' % model)
        
        loaded_model = load_model(curr_dir, _args['load_name'])
    elif isinstance(model, en.RandomForestRegressor):
        loaded_model = model
    else:
        raise ValueError('unknown model type: %s' % type(model))

    target_file = os.path.join(curr_dir, _args['data_dir'], 'valid', _args['filename'])

    if not os.path.exists(target_file):
        raise FileNotFoundError('The file specified does not exist')
    
    df = pd.read_csv(target_file)

    # pre-processing
    df = remove_outliers(df, _args)

    # split parameters from truth values
    y1 = pd.DataFrame(df.pop('feh'))
    x = df

    y2 = loaded_model.predict(x)
    y1 = np.array(y1.values.ravel())

    if _args['show']:
        truth_pred_side(x, y1, y2, _args)
        truth_pred_scat(y1, y2, _args)

    print('Score: %.6f' % loaded_model.score(x, y1))


# the function to use the model
def model_predict(x: pd.DataFrame, model: tp.Union[en.RandomForestRegressor, str], args: dict = {}) -> pd.DataFrame:
    '''
    Approximate metallicity values using photometric data as input
    
    Args:
        x (pd.DataFrame)                      : the input dataframe containing 4 photometric color channel magnitudes
        model (en.RandomForestRegressor | str): either the model itself, or the command to load the model
        args (dict)                           : customizable method arguments passed in as a dictionary
    
    Returns:
        pandas.DataFrame: corresponding metallicity value for each data point as approximated by model
    '''

    _args = {
        'load_name': '',
        'inplace'  : False
    }

    for key in args.keys():
        _args[key] = args[key]

    curr_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

    if _args['load_name'] == '':
        # try default load_name
        _args['load_name'] = default_savename

    if isinstance(model, str):
        if model != 'load':
            raise ValueError('unknown command: %s' % model)
        
        loaded_model = load_model(curr_dir, _args['load_name'])
    elif isinstance(model, en.RandomForestRegressor):
        loaded_model = model
    else:
        raise ValueError('unknown model type: %s' % type(model))

    y = loaded_model.predict(x)
    if _args['inplace']:
        new_df = x
    else:
        new_df = deepcopy(x)
    
    new_df['feh'] = y

    return new_df
