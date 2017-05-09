# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:41:53 2016

@author: Mathias

This file contains the functions that do the splitting process of the data.
"""

import os 
import sys
import time

import pandas as pd
import numpy as np
import random as rd



###############################################################################
### Get overview over the number of normal and anomaly samples ################
###############################################################################

def get_overview(data, label_col, values_norm=None, values_anom=None):
    """
    DESCRIPTION
    """
    
    levels_label = np.unique(data[label_col])
    
    if values_norm==None:
        values_norm = [val for val in levels_label if val not in values_anom]
    if values_anom==None:
        values_anom = [val for val in levels_label if val not in values_norm]
    
    label = data[label_col]
    
    print( '= Summary Dataset =\n' +
           '==================='   )
    
    
    ### get indices of anomalies
    print('Number of values for each anomaly level:')
    indicator_anom = label == ""
    # s = 0
    for val in values_anom:
        indicator_anom = np.logical_or( indicator_anom,  (label == val) )
        print '   ' + val + ': ' + repr(np.sum(label==val)) + ' values'

    (IDX_anom,) = np.where(indicator_anom.values)
    
    ### get indices of normal events
    print('\nNumber of values for each normal level:')
    indicator_norm = label == ""
    for val in values_norm:
        indicator_norm = np.logical_or( indicator_norm,  (label == val) )
        print '   ' + val + ': ' + repr(np.sum(label==val)) + ' values'

    (IDX_norm,) = np.where(indicator_norm.values)

    ratio = float(len(IDX_anom)) / float(len(IDX_norm))
    print( '\nOverview:\n' +
           '   normal samples:    ' + repr(len(IDX_norm)) + '\n' +
           '   anomalous samples: ' + repr(len(IDX_anom)) + '\n' +
           '   anom/norm ratio:   ' + repr(ratio) 
           + '\n' +
           '====================\n' )
 

###############################################################################

    
###############################################################################
### Create novelty dectection data split ######################################
###############################################################################

def NoveltySplit(data, label_col, values_norm=None, values_anom=None, 
                          n_test_samples=10000, n_train_samples_max = None, 
                          anom_ratio=0.1, seed=1):
    """
    DESCRIPTION
    """
    
    ### start clock
    start_time = time.time()
    
    ### if only values_norm or values_anom given define the other as the 
    ### complement in all label levels
    levels_label = np.unique(data[label_col])
    if values_norm==None:
        values_norm = [val for val in levels_label if val not in values_anom]
    if values_anom==None:
        values_anom = [val for val in levels_label if val not in values_norm]
        
    ### extract label column
    label = data[label_col]
    
    ### get indices of anomalies
    indicator_anom = label == ""
    for val in values_anom:
        indicator_anom = np.logical_or( indicator_anom,  (label == val) )
    ( IDX_anom, ) = np.where(indicator_anom.values)
    
    ### get indices of normal events
    indicator_norm = label == ""
    for val in values_norm:
        indicator_norm = np.logical_or( indicator_norm,  (label == val) )
    ( IDX_norm, ) = np.where(indicator_norm.values)

    
    ### define idx's of test instances
    n_test_norm = int(n_test_samples/(1+anom_ratio))
    n_test_anom = int(n_test_samples - n_test_norm)

    if n_test_anom > len(IDX_anom):
        raise Exception('[ERROR] Not enough anomaly samples in the dataset! [ERROR]' + '\n' +
                        repr(n_test_anom) + '>' +  repr(len(IDX_anom)) )
        
    rd.seed(seed)
    rd.shuffle(IDX_norm) # shuffle to break any ordering
    rd.shuffle(IDX_anom)
    IDX_test_anom = IDX_anom[:n_test_anom]
    IDX_test_norm = IDX_norm[:n_test_norm]
    IDX_anom = IDX_anom[n_test_anom:] # left over anomalies
    IDX_norm = IDX_norm[n_test_norm:] # left over normal instances

    ### define idx's of train instances
    if n_train_samples_max is None:
        IDX_train = IDX_norm
    else:
        if n_train_samples_max < len(IDX_norm):
            IDX_train = IDX_norm[:n_train_samples_max]
        else:
            IDX_train = IDX_norm # left over normal instances
    
    
    ### define train set, test set and test label
    train = data.drop(label_col, 1).loc[IDX_train,:]
    test = pd.concat( [ data.drop(label_col, 1).loc[IDX_test_norm,:],
                        data.drop(label_col, 1).loc[IDX_test_anom,:] ]  )
    label_test = np.hstack( [ np.repeat(0, len(IDX_test_norm) ), 
                              np.repeat(1, len(IDX_test_anom) ) ] )
    
    ### Print the result
    time_elapsed = round( time.time() - start_time, 1 ) # stop clock
    print( '= Summary: Novelty Split: =\n' +
           '===========================\n' +
           '   train set: ' + repr(len(IDX_train)) + '\n'
           '   - norm: ' + repr(len(IDX_train)) + '\n\n' +
           '   test set:  ' + repr(test.shape[0]) + '\n' +
           '   - norm: ' + repr(len(IDX_test_norm)) + '\n' + 
           '   - anom: ' + repr(len(IDX_test_anom)) + '\n\n' +
           'time elapsed: ' + repr(time_elapsed) + 's\n' +
           '===========================\n\n' )
    
    
    return train, test, label_test;

###############################################################################


###############################################################################
### Create outlier dectection data split ######################################
###############################################################################

def OutlierSplit(data, label_col, values_norm=None, values_anom=None, 
                          n_test_samples=10000, n_train_samples_max = None,
                          anom_ratio=0.1, 
                          anom_ratio_train=None, anom_ratio_test=None,
                          seed=1):
    """
    DESCRIPTION
    """
    
    ### start clock
    start_time = time.time()
    
    ### if only values_norm or values_anom given define the other as the 
    ### complement in all label levels
    levels_label = np.unique(data[label_col])
    if values_norm==None:
        values_norm = [val for val in levels_label if val not in values_anom]
    if values_anom==None:
        values_anom = [val for val in levels_label if val not in values_norm]
        
    ### if only anom_ratio_train or anom_ratio_test not given define as  
    ### anom_ratio
    if anom_ratio_train==None:
        anom_ratio_train = anom_ratio
    if anom_ratio_test==None:
        anom_ratio_test = anom_ratio
        
    ### extract label column
    label = data[label_col]
    
    ### get indices of anomalies
    indicator_anom = label == ""
    for val in values_anom:
        indicator_anom = np.logical_or( indicator_anom,  (label == val) )
    ( IDX_anom, ) = np.where(indicator_anom.values)
    data.from_records(data,indicator_anom) 
    
    ### get indices of normal events
    indicator_norm = label == ""
    for val in values_norm:
        indicator_norm = np.logical_or( indicator_norm,  (label == val) )
    ( IDX_norm, ) = np.where(indicator_norm.values)
    
    ### define idx's of test instances
    n_test_norm = int(n_test_samples/(1+anom_ratio_test))
    n_test_anom = int(n_test_samples - n_test_norm)
    rd.seed(seed)
    rd.shuffle(IDX_norm) # shuffle to break any ordering
    rd.shuffle(IDX_anom)
    IDX_test_anom = IDX_anom[:n_test_anom]
    IDX_test_norm = IDX_norm[:n_test_norm]
    IDX_anom = IDX_anom[n_test_anom:] # left over anomalies
    IDX_norm = IDX_norm[n_test_norm:] # left over normal instances

    ### define idx's of train instances
    if len(IDX_norm)*anom_ratio_train <= len(IDX_anom):
        n_train_norm = int( len(IDX_norm) )
        n_train_anom = int( len(IDX_norm)*anom_ratio_train )
    else:
        n_train_anom = int( len(IDX_anom) )
        n_train_norm = int( n_train_anom/anom_ratio_train )
    
    IDX_train_anom = IDX_anom[:n_train_anom]
    IDX_train_norm = IDX_norm[:n_train_norm]
    IDX_anom = IDX_anom[n_train_anom:] # left over anomalies
    IDX_norm = IDX_norm[n_train_norm:] # left over normal instances
    
    
    ### define train set, test set and test label
    train = pd.concat( [ data.drop(label_col, 1).loc[IDX_train_norm,:],
                         data.drop(label_col, 1).loc[IDX_train_anom,:] ]  )
    label_train = np.hstack( [ np.repeat(0, len(IDX_train_norm) ), 
                               np.repeat(1, len(IDX_train_anom) ) ] )
    
    test = pd.concat( [ data.drop(label_col, 1).loc[IDX_test_norm,:],
                        data.drop(label_col, 1).loc[IDX_test_anom,:] ]  )
    label_test = np.hstack( [ np.repeat(0, len(IDX_test_norm) ), 
                              np.repeat(1, len(IDX_test_anom) ) ] )
    
    ### shuffle train set and corresponding label
    permutation_train = np.random.permutation(train.shape[0])
    train.index = range(train.shape[0])
    train = train.loc[permutation_train,:]
    label_train = label_train[permutation_train]
    
    ### cut train if n_train_samples_max not none
    if ( (n_train_samples_max is not None) 
          and (n_train_samples_max < train.shape[0])):
        train = train[:n_train_samples_max]
        label_train = label_train[:n_train_samples_max]
    
    ### Print the result
    ratio_train_obs = float(len(IDX_train_anom)) / float(len(IDX_train_norm))
    ratio_test_obs = float(len(IDX_test_anom)) / float(len(IDX_test_norm))
    time_elapsed = round( time.time() - start_time, 1 ) # stop clock
    
    print( '= Summary: Outlier Split: =\n' +
           '===========================\n' +
           '   train set: ' + repr(train.shape[0]) + '\n'
           '   - norm: ' + repr(len(label_train)-np.sum(label_train)) + '\n' +
           '   - anom: ' + repr(np.sum(label_train)) + '\n' +
           '   (ratio: ' + repr(ratio_train_obs) 
           + ')\n\n' +
           '   test set:  ' + repr(test.shape[0]) + '\n' +
           '   - norm: ' + repr(len(IDX_test_norm)) + '\n' + 
           '   - anom: ' + repr(len(IDX_test_anom)) + '\n' +
           '   (ratio: ' + repr(ratio_test_obs) 
           + ')\n\n' +
           'time elapsed: ' + repr(time_elapsed) + 's\n' +
           '===========================\n\n' )
    
    
    return train, test, label_test;

###############################################################################




"""

### TESTING ONLY !!!!
def load_data():
    global data
    
    data = pd.read_csv("D:/Daten/ETH/ETH 5.1/Masterarbeit/programming/Test/KDD Cup 1999/data/kddcup.data.corrected.csv")
    test = pd.read_csv("D:/Daten/ETH/ETH 5.1/Masterarbeit/programming/Test/KDD Cup 1999/data/test.csv")
    data.columns = list(test.columns) + ['label']

load_data()

### for KDD Cup dataset
label_col= ['label']
values_norm = ['normal.']
unique = np.unique(data[label_col])
values_anom = [col for col in unique if not col in values_norm]


# get_overview(data, label_col=label_col, values_norm=values_norm)
# train, test, label_test = NoveltySplit(data, label_col=label_col, values_norm=values_norm, 
#                           n_test_samples=11000, anom_ratio=0.1)

train, test, label_test = OutlierSplit(
                             data, label_col=label_col, 
                             values_norm=values_norm, 
                             n_test_samples=11000, 
                             anom_ratio=0.1)

# %reset

"""