#!/usr/bin/env python

'''
    Photometric Approximator of Stellar Metallicity (c) by Rik Ghosh, Soham Saha

    Photometric Approximator of Stellar Metallicity is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this work.
    If not, see https://creativecommons.org/licenses/by/4.0
'''

'''
    This python script is used to visualize some of the figures that can serve as
    visual affirmation metrics for guaging the accuracy of the supervised machine
    learning model

    This file should not need any further editing from the side of the user
'''

# standard library imports
import os

# third-party imports
import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np

COLORS = ['ug', 'gr', 'ri', 'iz']   # expected colors in the input dataframe

# produce a before-after comparison box-and-whisker plot for the preprocessing
def boxplot(before_df: pd.DataFrame, after_df: pd.DataFrame, args: dict) -> None:
    '''
    A box-and-whisker plot that shows a before preprocessing and after preprocessing for
    the four color channels in the dataframe
    
    Args:
        before_df (pandas.DataFrame): input dataframe before preprocessing
        after_df (pandas.DataFrame) : input dataframe after preprocessing
        args (dict)                 : customizable method arguments passed in as a dictionary
        
    Returns:
        None
    '''

    # base arguments dictionary -> modified by the args parameter
    _args = {
        'iqr_factor' : 1.5,         # factor for scaling whiskers for min and max thresholds
        'save'       : False,       # save the figure
        'dpi'        : 300          # save quality
    }

    for key in args.keys():
        _args[key] = args[key]      # update parameters

    fig, axs = plt.subplots(1, 2, sharey='all')

    axs[0].boxplot(before_df, vert=False, whis=_args['iqr_factor'], labels=COLORS)
    axs[0].set_title('Before Pre-processing Data')
    axs[0].set_xlabel('Color Magnitude')
    axs[0].set_ylabel('Colors')

    axs[1].boxplot(after_df, vert=False, whis=_args['iqr_factor'], labels=COLORS)
    axs[1].set_title('After Pre-processing Data')
    axs[1].set_xlabel('Color Magnitude')

    fig.suptitle('Result of Processing Input Data')

    if _args['save']:
        # join path to target directory
        curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dir_path = os.path.join(curr_dir, 'figures')                # directory for figures
        
        if not os.path.exists(dir_path):                            # check to see if directory already exists
            os.mkdir(dir_path)
        
        target_path = os.path.join(dir_path, 'boxplot.png')
        plt.savefig(target_path, dpi=_args['dpi'])
        print('Saved figures/boxplot.png')

    plt.show()
    plt.close()


# produce a side-by-side comparison of the ground truth values and the predicted values against each color
def truth_pred_side(x: list, y1: list, y2: list, args: dict = {}) -> None:
    '''
    A side-by-side comparison of the ground truth values and the model predictions for each color used
    in the training of the model
    
    Args:
        x (list)   : the training parameter
        y1 (list)  : the ground-truth values
        y2 (list)  : the model predictions
        args (dict): customizable method arguments passed in as a dictionary
        
    Returns:
        None
    '''

    _args = {
        'colors' : [],              # colors for each group
        'save'   : False,           # save the figure
        'dpi'    : 300              # save quality
    }

    for key in args.keys():
        _args[key] = args[key]      # update parameters

    if len(_args['colors']) < 2:
        # use default colors
        _args['colors'] += ['firebrick', 'skyblue']

    fig, axs = plt.subplots(4, 2, sharex='row')
    plt.tight_layout()

    for i in range(len(COLORS)):
        col = COLORS[i]
        label = '%s-%s' % (col[0], col[1])

        for j in range(2):
            if j % 2 == 0:
                axs[i, j].scatter(x[col], y1, color=_args['colors'][0], marker='.')
                axs[i, j].set_title('True $%s$' % label.upper())
                axs[i, j].legend(['True Value'])
            else:
                axs[i, j].scatter(x[col], y2, color=_args['colors'][1], marker='.')
                axs[i, j].set_title('Predicted $%s$' % label.upper())
                axs[i, j].legend(['Predicted Value'])

            axs[i, j].set_xlabel('$%s$' % label)
            axs[i, j].set_ylabel('Metallicity $[Fe/H]$')

    fig.suptitle('Ground Truths and Model Predictions')
    fig.supxlabel('Side-by-Side comparison')

    if _args['save']:
        # join path to target directory
        curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dir_path = os.path.join(curr_dir, 'figures')                # directory for figures
        
        if not os.path.exists(dir_path):                            # check to see if directory already exists
            os.mkdir(dir_path)
        
        target_path = os.path.join(dir_path, 'comp_side.png')
        plt.savefig(target_path, dpi=_args['dpi'])
        print('Saved figures/comp_side.png')

    plt.show()
    plt.close()
        

# produce a scatter relationship for the ground-truths and model predictions to visualize associations
def truth_pred_scat(y1: list, y2: list, args: dict = {}) -> None:
    '''
    A scatter plot to view the associations between the ground truth values and the model predictions. This
    serves as a visual metric to guage the accuracy of the model
    
    Args:
        y1 (list)   : the ground-truth values
        y2 (list)   : the model predictions
        args (dict) : customizable method arguments passed in as a dictionary
        
    Returns:
        None
    '''

    _args = {
        'save': False,          # save the figure
        'dpi' : 300             # save quality
    }

    for key in args.keys():
        _args[key] = args[key]  # update parameters

    tp = np.vstack((y1, y2))
    z = sts.gaussian_kde(tp)(tp)

    plt.scatter(y1, y2, c=z, marker='.')
    plt.plot(y1, y1, 'r-', label='One-to-one Regression Line')
    plt.xlabel(r'$[Fe/H]_{SSPP}$')
    plt.ylabel(r'$[Fe/H]_{RF}$')
    plt.colorbar(label='Scatter Density')
    plt.legend(loc='best')
    
    plt.suptitle('Model Predictions vs Ground Truth Scatter Plot')
    plt.title('Density Profile')

    if _args['save']:
        # join path to target directory
        curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dir_path = os.path.join(curr_dir, 'figures')                # directory for figures
        
        if not os.path.exists(dir_path):                            # check to see if directory already exists
            os.mkdir(dir_path)
        
        target_path = os.path.join(dir_path, 'scat.png')
        plt.savefig(target_path, dpi=_args['dpi'])
        print('Saved figures/scat.png')

    plt.show()
    plt.close()
