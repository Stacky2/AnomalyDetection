# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 11:39:11 2017

@author: Mathias

This file contains the class definition and functions of the AnomalyDetaSet.
The functions also allows to generate data splits for Anomaly Detection out of 
the raw data, using the function "getAnomalySplit".
"""

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import ast
import os 
import time
import math

from sklearn.manifold import TSNE

### define directories
home_dir = os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir)
data_dir = os.path.join(home_dir, "data")



###############################################################################
### Anomaly Dataset Class #####################################################
###############################################################################

class AnomalyDataSet():
    """Anomaly dataset class"""
    dataset                 = None
    label_col               = None
    values_norm             = None
    values_anom             = None
    mode                    = None
    n_test_samples          = None
    n_train_samples_max     = None    
    anom_ratio              = None
    anom_ratio_train        = None
    anom_ratio_test         = None
    n_noise_cols            = 0
    seed                    = None
    
    def __init__(self, dataset, mode):
        """Initializes an AnomalyDataSet."""
        self.dataset = dataset
        self.mode   = mode
        
    def set_params(self, **params):
        """Sets all the variables of an instance according to the key-value 
        pairs given in the dictionary **params."""
        for par in params:
            setattr(self, par, params[par])
            
    def set_params_by_config(self, config):
        """Sets all the variables of an instance according to the section in 
        the configfile "config" in section '["self.dataset"]'. """
        params = {}
        for param in config[self.dataset].keys():
            params[str(param)] = ast.literal_eval(config[self.dataset][param])
            
        self.set_params(**params)
        return params    
    
    def load_data_from_csv(self, csv_dir):
        """Loads the dataset with the name "self.dataframe" from the path
        "../data/[self.dataset]/data.csv". """
        self.data = pd.read_csv(csv_dir, sep=',')
        ### define categorical features
        for col in self.fac_cols:
            self.data[col] = pd.Categorical(self.data[col])
        self.data[self.label_col] = pd.Categorical(self.data[self.label_col])
    
    def load_data_from_folder(self, home_dir):
        """Loads the dataset with the name "data.csv" in the directory 
        home_dir"."""
        dataset_dir = os.path.join(home_dir, "data", self.dataset)
        csv_dir = os.path.join(dataset_dir, "data.csv")
    
        self.load_data_from_csv(csv_dir)
        
        self.add_noise()
        #self.add_gauss_noise()
        #self.add_cat_noise()
        
    def add_noise(self):
        """Adds noise features to the raw dataset."""
        ncols_org = self.data.shape[1] 
        ratio_cat = float(len(self.fac_cols))/ float(ncols_org)
                       
        ncols_cat = int(math.floor(ratio_cat * self.n_noise_cols))
        ncols_num = int(math.ceil((1-ratio_cat) * self.n_noise_cols))
        
        self.add_gauss_noise(ncols = ncols_num)
        self.add_cat_noise(ncols = ncols_cat)
        
    def add_gauss_noise(self, ncols):
        """Adds a number of ncols gaussian noise features to the raw 
        dataset."""
        np.random.seed(seed=self.seed)
        noise_features = pd.DataFrame(
                data=np.random.normal(size=(self.data.shape[0], ncols)), 
                columns=['num_noise'+repr(i) for i in range(ncols)] 
                )
        self.data = pd.concat([self.data,noise_features], axis=1)
        if ncols > 1:
            print ( repr(ncols) 
            + " standard normal noise features added to original data." )
            
    def add_cat_noise(self, ncols):
        """Adds a number o ncols categorical noise features to the raw 
        dataset."""
        np.random.seed(seed=self.seed)
        noise_features = pd.DataFrame(
                data=np.random.choice(['A','B','C'], 
                                      size=(self.data.shape[0], ncols)), 
                columns=['cat_noise'+repr(i) for i in range(ncols)]
                )
        for col in noise_features:
            noise_features[col] = pd.Categorical(noise_features[col])
            
        self.data = pd.concat([self.data, noise_features], axis=1)
        self.fac_cols.append(noise_features.columns)
        if ncols > 1:
            print ( repr(ncols) 
            + " categorical noise features added to original data." )

    def get_overview(self):
        """
        Prints an overview of the raw data set with information as number of 
        instances per level, etc.
        """
        
        levels_label = np.unique(self.data[self.label_col])
        
        if self.values_norm==None:
            self.values_norm = [val for val in levels_label 
                                if val not in self.values_anom]
        if self.values_anom==None:
            self.values_anom = [val for val in levels_label 
                                if val not in self.values_norm]
        
        label = self.data[self.label_col]
        
        self.overview = ( '= Summary Dataset =\n' +
                          '===================\n'   )
        
        
        ### get indices of anomalies
        self.overview = self.overview + ('Number of values for each anomaly'+ 
                                         'level:\n')
        indicator_anom = label == ""
        # s = 0
        for val in self.values_anom:
            indicator_anom = np.logical_or( indicator_anom,  (label == val) )
            self.overview = (self.overview + '   ' + val + ': ' 
                             + repr(np.sum(label==val)) + ' values\n')
    
        (IDX_anom,) = np.where(indicator_anom.values)
        
        ### get indices of normal events
        self.overview = (self.overview 
                         + '\nNumber of values for each normal level:\n')
        indicator_norm = label == ""
        for val in self.values_norm:
            indicator_norm = np.logical_or( indicator_norm,  (label == val) )
            self.overview = (self.overview + '   ' + val + ': ' 
                             + repr(np.sum(label==val)) + ' values\n')
    
        (IDX_norm,) = np.where(indicator_norm.values)
    
        ratio = float(len(IDX_anom)) / float(len(IDX_norm))
        self.overview = self.overview + ( '\nOverview:\n' +
               '   normal samples:    ' + repr(len(IDX_norm)) + '\n' +
               '   anomalous samples: ' + repr(len(IDX_anom)) + '\n' +
               '   anom/norm ratio:   ' + repr(ratio) + '\n' +
               '====================\n' )
        
        print self.overview
        return self.overview;
    
        
    def plot(self, nsamples=100, perplexity=30.0, n_iter=1000, seed=1):
        """
        Plots a t-SNE plot of the dataset with different colours indicating
        normal and anomaly samples.
        """
        start_time = time.time() # start clock
        
        ### select part of data and label 
        rd.seed(seed)
        X_copy = self.data.copy()
        X_copy.drop(self.label_col, axis=1)
        X_copy.index = range(X_copy.shape[0])
        IDX_sample = rd.sample( range(X_copy.shape[0]), 
                                k=min(nsamples,X_copy.shape[0]) )
        Xsel = pd.get_dummies(X_copy.loc[IDX_sample,:])
        label = self.data.loc[:,self.label_col]
        label_sel = label[IDX_sample]
        label_values = np.unique(label)
        
        ### fit TSNE model
        model = TSNE(n_components=2,  perplexity=perplexity, n_iter=n_iter, 
                     random_state=seed, verbose=1)
        np.set_printoptions(suppress=True)
        X_TSNE = model.fit_transform( Xsel ) 
        
        ### plot TSNE with col as colour label
        plt.figure()
        plt.figure(figsize=(15, 10)) 
        col_list = ['coral','steelblue','aquamarine','gold',
                    'yellowgreen','fuchsia','orchid'] # add more
        
        for i in range(len(label_values)):
            color = col_list[i]
            print color    
            plt.scatter( X_TSNE[np.where(label_sel==label_values[i]),0], 
                         X_TSNE[np.where(label_sel==label_values[i]),1], 
                         #marker='x', 
                         color=color,
                         linewidth='1', alpha=0.8, label=label_values[i] )
        
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        #plt.title('T-SNE on ' + repr(nsamples) + ' train samples')
        plt.legend(loc='best')
        plt.savefig(os.path.join(os.path.join(data_dir,self.dataset),
                                 "TSNE.png"))
        #plt.savefig('../data/'+self.dataset+'/TSNE.png')
        plt.show()
        
        time_elapsed = round(time.time() - start_time, 1) # stop clock
        print ('  model fit finished! (time elapsed: ' 
                + repr(time_elapsed) + 's)\n')
            

### Create novelty dectection data split ######################################
###############################################################################

    def getNoveltySplit(self, print_overview=False):
        """
        This function generates a novelty split of the data, i.e. train set,
        test set and anomaly indicator label for the test set. Note that the 
        train set consists of only the normal class whereas the test set 
        consits of normal an anomaly samples.
        """
        
        if print_overview: 
            self.get_overview()
        
        ### start clock
        start_time = time.time()
        
        ### if only values_norm or values_anom given define the other as the 
        ### complement in all label levels
        levels_label = np.unique(self.data[self.label_col])
        if self.values_norm==None:
            self.values_norm = [val for val in levels_label 
                                if val not in self.values_anom]
        if self.values_anom==None:
            self.values_anom = [val for val in levels_label 
                                if val not in self.values_norm]
            
        ### extract label column
        label = self.data[self.label_col]
        
        ### get indices of anomalies
        indicator_anom = label == ""
        for val in self.values_anom:
            indicator_anom = np.logical_or( indicator_anom,  (label == val) )
        ( IDX_anom, ) = np.where(indicator_anom.values)
        
        ### get indices of normal events
        indicator_norm = label == ""
        for val in self.values_norm:
            indicator_norm = np.logical_or( indicator_norm,  (label == val) )
        ( IDX_norm, ) = np.where(indicator_norm.values)
    
        
        ### define idx's of test instances
        n_test_norm = int(self.n_test_samples/(1+self.anom_ratio))
        n_test_anom = int(self.n_test_samples - n_test_norm)
    
        if n_test_anom > len(IDX_anom):
            raise Exception(
               '[ERROR] Not enough anomaly samples in the dataset! [ERROR]' 
                + '\n' + repr(n_test_anom) + '>' +  repr(len(IDX_anom)) )
            
        rd.seed(self.seed)
        rd.shuffle(IDX_norm) # shuffle to break any ordering
        rd.shuffle(IDX_anom)
        IDX_test_anom = IDX_anom[:n_test_anom]
        IDX_test_norm = IDX_norm[:n_test_norm]
        IDX_anom = IDX_anom[n_test_anom:] # left over anomalies
        IDX_norm = IDX_norm[n_test_norm:] # left over normal instances
    
        ### define idx's of train instances
        if self.n_train_samples_max is None:
            IDX_train = IDX_norm
        else:
            if self.n_train_samples_max < len(IDX_norm):
                IDX_train = IDX_norm[:self.n_train_samples_max]
            else:
                IDX_train = IDX_norm # left over normal instances
        
        
        ### define train set, test set and test label
        train = self.data.drop(self.label_col, 1).loc[IDX_train,:]
        test = pd.concat( 
                [ self.data.drop(self.label_col, 1).loc[IDX_test_norm,:],
                  self.data.drop(self.label_col, 1).loc[IDX_test_anom,:] ]  
                )
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
        

### Create outlier dectection data split ######################################
###############################################################################

    def getOutlierSplit(self, print_overview=False):
        """
        This function generates a outlier split of the data, i.e. train set,
        test set and anomaly indicator label for the test set. Note that the 
        train set and the test set consist of normal and anomaly samples.
        """
        if print_overview: 
            self.get_overview()
        
        ### start clock
        start_time = time.time()
        
        ### if only values_norm or values_anom given define the other as the 
        ### complement in all label levels
        levels_label = np.unique(self.data[self.label_col])
        if self.values_norm==None:
            self.values_norm = [val for val in levels_label 
                                if val not in self.values_anom]
        if self.values_anom==None:
            self.values_anom = [val for val in levels_label 
                                if val not in self.values_norm]
            
        ### if only anom_ratio_train or anom_ratio_test not given define as  
        ### anom_ratio
        if self.anom_ratio_train==None:
            self.anom_ratio_train = self.anom_ratio
        if self.anom_ratio_test==None:
            self.anom_ratio_test = self.anom_ratio
            
        ### extract label column
        label = self.data[self.label_col]
        
        ### get indices of anomalies
        indicator_anom = label == ""
        for val in self.values_anom:
            indicator_anom = np.logical_or( indicator_anom,  (label == val) )
        ( IDX_anom, ) = np.where(indicator_anom.values)
        self.data.from_records(self.data,indicator_anom) 
        
        ### get indices of normal events
        indicator_norm = label == ""
        for val in self.values_norm:
            indicator_norm = np.logical_or( indicator_norm,  (label == val) )
        ( IDX_norm, ) = np.where(indicator_norm.values)
        
        ### define idx's of test instances
        n_test_norm = int(self.n_test_samples/(1+self.anom_ratio_test))
        n_test_anom = int(self.n_test_samples - n_test_norm)
        rd.seed(self.seed)
        rd.shuffle(IDX_norm) # shuffle to break any ordering
        rd.shuffle(IDX_anom)
        IDX_test_anom = IDX_anom[:n_test_anom]
        IDX_test_norm = IDX_norm[:n_test_norm]
        IDX_anom = IDX_anom[n_test_anom:] # left over anomalies
        IDX_norm = IDX_norm[n_test_norm:] # left over normal instances
    
        ### define idx's of train instances
        if len(IDX_norm)*self.anom_ratio_train <= len(IDX_anom):
            n_train_norm = int( len(IDX_norm) )
            n_train_anom = int( len(IDX_norm)*self.anom_ratio_train )
        else:
            n_train_anom = int( len(IDX_anom) )
            n_train_norm = int( n_train_anom/self.anom_ratio_train )
        
        IDX_train_anom = IDX_anom[:n_train_anom]
        IDX_train_norm = IDX_norm[:n_train_norm]
        IDX_anom = IDX_anom[n_train_anom:] # left over anomalies
        IDX_norm = IDX_norm[n_train_norm:] # left over normal instances
        
        
        ### define train set, test set and test label
        train = pd.concat(
                   [ self.data.drop(self.label_col, 1).loc[IDX_train_norm,:],
                     self.data.drop(self.label_col, 1).loc[IDX_train_anom,:] ]  
                   )
        label_train = np.hstack( [ np.repeat(0, len(IDX_train_norm) ), 
                                   np.repeat(1, len(IDX_train_anom) ) ] )
        
        test = pd.concat( 
                  [ self.data.drop(self.label_col, 1).loc[IDX_test_norm,:],
                    self.data.drop(self.label_col, 1).loc[IDX_test_anom,:] ]  
                  )
        label_test = np.hstack( [ np.repeat(0, len(IDX_test_norm) ), 
                                  np.repeat(1, len(IDX_test_anom) ) ] )
        
        ### shuffle train set and corresponding label
        permutation_train = np.random.permutation(train.shape[0])
        train.index = range(train.shape[0])
        train = train.loc[permutation_train,:]
        label_train = label_train[permutation_train]
        
        ### cut train if n_train_samples_max not none
        if ( (self.n_train_samples_max is not None) 
              and (self.n_train_samples_max < train.shape[0])):
            train = train[:self.n_train_samples_max]
            label_train = label_train[:self.n_train_samples_max]
        
        ### Print the result
        ratio_train_obs = ( float(len(IDX_train_anom)) 
                            / float(len(IDX_train_norm)) )
        ratio_test_obs =  ( float(len(IDX_test_anom)) 
                            / float(len(IDX_test_norm)) )
        time_elapsed = round( time.time() - start_time, 1 ) # stop clock
        
        print( '= Summary: Outlier Split: =\n' +
               '===========================\n' +
               '   train set: ' + repr(train.shape[0]) + '\n'
               '   - norm: ' + repr(len(label_train)-np.sum(label_train)) 
               + '\n' +
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


### Create data split #########################################################
###############################################################################

    def getAnomalySplit(self, print_overview=False):
        """This function is just a wrapper of the functions getNoveltySplit
        and getOutlierSplit, where the choice which one is used is made by the 
        mode class variable."""
        
        if self.mode == "Novelty":
            train, test, label_test = self.getNoveltySplit(
                                       print_overview = print_overview )
            return train, test, label_test;
        elif self.mode == "Outlier":
            train, test, label_test = self.getOutlierSplit(
                                       print_overview = print_overview )
            return train, test, label_test;
        else:
            raise Exception( 'Please choose "mode" as one of "Novelty" or ' +
                             '"Outlier".')

###############################################################################
###############################################################################

"""
### For test purposes only!
if __name__ == "__main__":
    model = AnomalyDataSet(dataset="Forest Cover Type", mode = "Novelty")
    model.set_params_by_config(config_data)
    model.load_data_from_folder(home_dir)
    model.get_overview()
    train, test, label_test = model.getAnomalySplit()
"""
