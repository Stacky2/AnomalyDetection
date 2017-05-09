# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:14:02 2016

@author: Mathias

= Representation Models =
=========================

This file contains the class definitions and functions of the following
Representation Models:
    - Identity model
    - Autoencoder model
    - Principal Component analysis model
    - Fast independent component model
    - Truncated singular value model
    - Entity Embedding model
    
"""

import os
import h2o
import time
import configparser
import ast
import sys

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random as rd

from sklearn.manifold import LocallyLinearEmbedding
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from sklearn import decomposition
from sklearn.manifold import Isomap

### define max_mem_size allwoed for h2o (in GB, choose none for laptop)
max_mem_size = 16

### define directories
RepModels_dir = os.path.dirname(__file__)
home_dir            = os.path.normpath(RepModels_dir 
                                           + os.sep + os.pardir)
AnomalyModels_dir   = os.path.join(home_dir, "AnomalyModels")

### import own modules
## import the AnomalyModel classes
sys.path.insert(0, AnomalyModels_dir)
from AnomalyModels import anomaly_sampler

### config parser for representation models 
config_Rep = configparser.ConfigParser()
config_Rep.read(os.path.join(RepModels_dir,'config.ini'))




###############################################################################
### Base class Representation Model ###########################################
###############################################################################

class RepModel:
    """Base class for representation models"""
    def __init__(self, ID, model_type, model_pars):
        self.model              = None
        self.model_type         = model_type
        self.ID                 = ID
        self.model_pars         = model_pars
        self.n_features         = None
        self.verbose            = None          # verbosity of output
        self.fit_time           = None          # elapsed time for fit
        self.trafo_time         = None          # elapsed time for test set 
                                                # trafo
                                                
    def set_params(self, **params):
        for par in params:
            setattr(self, par, params[par])
            
    def get_representation( self, data, test=False ):
        ### start clock
        start_time = time.time()
        
        self.print_sample_computation(data)
        
        data_rep = self.compute_representation(data)
        
        self.save_test_trafo_time(start_time, test)
                           
        return data_rep;
            

    def print_fit_started(self):
        if self.verbose == 1:
            print ( '> Fit ' + self.model_type + '...' )
            for i in range(len(self.model_pars)):
                par = self.model_pars[i]
                print( '     ' + (par + 20*' ')[:20] 
                       + '= ' + repr(getattr(self, par)) )

    def model_fit_finished(self, start_time):
        time_elapsed = round(time.time() - start_time, 1)
        self.fit_time = time_elapsed
        if self.verbose == 1:
            print ('  model fit finished! (time elapsed: ' 
                    + repr(self.fit_time) + 's)\n') 
            
    def print_sample_computation(self, data):
        if self.verbose == 1:
            print ( '> Compute "' + self.ID + '" representations ('
                     + repr(data.shape[0]) + ' samples) ...' )
            
    def save_test_trafo_time(self, start_time, test):
        if test:
            time_elapsed = round(time.time() - start_time, 1)
            self.trafo_time = time_elapsed

###############################################################################


###############################################################################
### get default parameters from config file ###################################
###############################################################################

def set_pars_by_config(self, config):
    for key in config[self.ID]:
        if not hasattr(self, key) or (getattr(self, key) is None):
            setattr(self, key, ast.literal_eval(config[self.ID][key]))

###############################################################################


###############################################################################
### Identity representation model #############################################
###############################################################################

class ID_RepModel(RepModel):
    """Identity representation models"""
    def __init__( self, verbose=None ):
        """Initialization of the representation models"""
        RepModel.__init__( self, 
                           ID = 'ID',
                           model_type = 'Identity representation model', 
                           model_pars = [] )
        self.verbose = verbose

        
        set_pars_by_config(self, config_Rep)
        
    def fit(self, train):
        """Fits the representation model"""
        self.n_features = train.shape[1]
        
        ### start clock
        start_time = time.time()
        
        self.print_fit_started()
        
        ### stop clock & save fit time
        self.model_fit_finished(start_time)
        
    def compute_representation(self, data):
        """A function that computes concrete representation of the samples."""
        ### compute representation
        data_ID = data.copy()
        data_ID.index = range(data_ID.shape[0])
        data_ID.columns = ['ID'+repr(i) for i in range(data_ID.shape[1])]
        return data_ID;
    
###############################################################################


###############################################################################
### Autoencoder representation model ##########################################
###############################################################################

class AE_RepModel(RepModel):
    """
    DESCRIPTION
    """
    
    def __init__( self,
                  hidden    = None, 
                  rep_layer = None, 
                  epochs    = None, 
                  ncores    = None, 
                  seed      = None,
                  verbose   = None ):
        """
        DESCRIPTION
        """
        RepModel.__init__( self, 
                           ID = 'AE',
                           model_type = 'Autoencoder representation model', 
                           model_pars = [ 'factor','hidden',
                                          'rep_layer','epochs'] )
        
        self.hidden     = hidden
        self.rep_layer  = rep_layer
        self.epochs     = epochs
        self.ncores     = ncores
        self.seed       = seed
        self.verbose    = verbose

        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        
        ### start clock
        start_time = time.time()
        
        ### Compute default hidden architecture if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
        
        if (self.hidden is None) and (self.rep_layer is not None):
            raise Exception('It is not allowed to define "rep_layer" without\
            defining "hidden"!')
        if not ( hasattr(self, 'hidden') 
                 & (self.hidden is not None)):
            self.hidden = 3*[int(self.factor*dim_dummy)]
        if not ( hasattr(self, 'rep_layer') 
                 & (self.rep_layer is not None)):
            self.rep_layer = 1
            
        self.n_features = self.hidden[self.rep_layer]
            
        self.print_fit_started()
        
        ### initialize H2O cluster
        h2o.init(nthreads=self.ncores, max_mem_size=max_mem_size)
        train_h2o = h2o.H2OFrame(train) # save train as H2OFrame
        
        ### fit model
        self.model = H2OAutoEncoderEstimator(
                        activation      = "Tanh",
                        hidden          = self.hidden,
                        epochs          = self.epochs,
                        score_interval  = 1,
                        stopping_rounds = 0,
                        overwrite_with_best_model = True,
                        seed = self.seed )
        self.model.train(training_frame=train_h2o) 
        
        ### print that model fit finished
        self.model_fit_finished(start_time)
        
    def compute_representation(self, data):
        ### compute representation
        data_H2O = h2o.H2OFrame(data)
        data_AE = self.model\
                      .deepfeatures(data_H2O, self.rep_layer).as_data_frame()
        
        ### adjust row and column indexes
        data_AE.index = range(data_AE.shape[0])
        data_AE.columns = ['AE'+repr(i) for i in range(data_AE.shape[1])]
        return data_AE;

###############################################################################


###############################################################################
### PCA representation model ##################################################
###############################################################################

class PCA_RepModel(RepModel):
    """
    DESCRIPTION
    """
    
    def __init__( self,
                  n_components   = None, 
                  seed           = None, 
                  verbose        = None  ):
        """
        DESCRIPTION
        """
        RepModel.__init__( self, 
                           ID = 'PCA',
                           model_type = ( 'Principal Component ' +
                                          'representation model' ), 
                           model_pars = ['factor','n_components'] )
        
        self.n_components   = n_components
        self.seed           = seed
        self.verbose        = verbose
        
        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        ### Compute default n_components if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
        if not ( hasattr(self, 'n_components') 
                 & (self.n_components is not None)):
            self.n_components = int(self.factor*dim_dummy)
        
        self.n_features = self.n_components
        
        ### start clock
        start_time = time.time()
        
        self.print_fit_started()
        
        # PCA, IncrementalPCA, NMF, FastICA, MiniBatchSparsePCA
        self.model = decomposition.PCA( n_components = self.n_components,
                                        random_state = self.seed )
        
        self.model.fit(X=pd.get_dummies(train)) 
        
        ### stop clock
        self.model_fit_finished(start_time)
        
    def compute_representation(self,data):
        ### compute representation model
        data_PCA = pd.DataFrame( self.model.transform(pd.get_dummies(data)) )
        
        ### adjust row and column indexes
        data_PCA.index = range(data_PCA.shape[0])
        data_PCA.columns = ['PCA'+repr(i) for i in range(data_PCA.shape[1])]
        return data_PCA;
        
###############################################################################


###############################################################################
### FICA representation model ##################################################
###############################################################################

class FICA_RepModel(RepModel):
    """
    DESCRIPTION
    """
    
    def __init__( self, 
                  n_components  = None, 
                  seed          = None, 
                  verbose       = None  ):
        """
        DESCRIPTION
        """
        RepModel.__init__( self, 
                           ID = 'FICA',
                           model_type = ( 'Fast Independent Components ' +
                                          'representation model' ), 
                           model_pars = ['factor','n_components'] )
        self.n_components   = n_components
        self.seed           = seed
        self.verbose        = verbose
        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        ### Compute default n_components if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
        
        if not ( hasattr(self, 'n_components') 
                 & (self.n_components is not None)):
            self.n_components = int(self.factor*dim_dummy)

        self.n_features = self.n_components
            
        ### start clock
        start_time = time.time()
        
        self.print_fit_started()
        
        # PCA, IncrementalPCA, NMF, FastICA, MiniBatchSparsePCA
        self.model = decomposition.FastICA( n_components=self.n_components, 
                                        random_state=self.seed )
        self.model.fit(X=pd.get_dummies(train)) 
        
        ### stop clock
        time_elapsed = round(time.time() - start_time, 1) 
        self.fit_time = time_elapsed
        
        self.model_fit_finished(start_time)
        
    def compute_representation(self, data):
        data_FICA = pd.DataFrame( self.model.transform(pd.get_dummies(data)) )
        data_FICA.index = range(data_FICA.shape[0])
        data_FICA.columns = ['FICA'+repr(i) for i in range(data_FICA.shape[1])]
        return data_FICA;

###############################################################################


###############################################################################
### TSVD representation model ##################################################
###############################################################################

class TSVD_RepModel(RepModel):
    """
    DESCRIPTION
    """
    
    def __init__( self, 
                  n_components  = None, 
                  seed          = None, 
                  verbose       = None  ):
        """
        DESCRIPTION
        """
        RepModel.__init__( self, 
                           ID = 'TSVD',
                           model_type = ( 'Truncated Singular Value ' +
                                          'representation model' ), 
                           model_pars = ['factor','n_components'] )
        
        self.n_components   = n_components
        self.seed           = seed
        self.verbose        = verbose
        
        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        
        ### Compute default n_components if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
        if not ( hasattr(self, 'n_components') 
                 & (self.n_components is not None)):
            self.n_components = int(self.factor*dim_dummy)
        
        self.n_features = self.n_components
        
        ### start clock
        start_time = time.time()
        
        self.print_fit_started()
        
        # PCA, IncrementalPCA, TSVD, FastICA, MiniBatchSparsePCA
        self.model = decomposition.TruncatedSVD( n_components=self.n_components , 
                                       random_state=self.seed )
        
        self.model.fit(X=pd.get_dummies(train)) 
        
        ### stop clock
        self.model_fit_finished(start_time)
        
    def compute_representation(self, data):
        data_TSVD = pd.DataFrame( self.model.transform(pd.get_dummies(data)) )
        data_TSVD.index = range(data_TSVD.shape[0])
        data_TSVD.columns = ['TSVD'+repr(i) for i in range(data_TSVD.shape[1])]
        return data_TSVD;
    
###############################################################################


###############################################################################
### EE representation model ##################################################
###############################################################################

class EE_RepModel(RepModel):
    """
    DESCRIPTION
    """
    
    def __init__( self, 
                  factor       = None,
                  n_components = None,
                  hidden       = None, 
                  depth        = None,
                  rep_layer    = None, 
                  epochs       = None, 
                  ncores       = None, 
                  seed         = None,
                  verbose      = None  ):
        """
        DESCRIPTION
        """
        RepModel.__init__( self, 
                           ID = 'EE',
                           model_type = ( 'Entity Embedding ' +
                                          'representation model' ), 
                           model_pars = [ 'factor','hidden',
                                          'rep_layer','epochs'] )
    
        self.factor         = factor
        self.n_components   = n_components
        self.hidden         = hidden
        self.depth          = depth
        self.rep_layer      = rep_layer
        self.epochs         = epochs
        self.ncores         = ncores
        self.seed           = seed
        self.verbose        = verbose
        
        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        
        ### Compute default n_components if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
        if (self.hidden is not None) and (self.n_components is not None):
            raise Exception( 'Error: You are not allowed to specify both '+
                             '"hidden" and "n_components" at the same time!' )
            
        if (self.factor is not None) and (self.n_components is not None):
            raise Exception( 'Error: You are not allowed to specify both '+
                             '"factor" and "n_components" at the same time!' )
            
        if (self.factor is not None) and (self.hidden is not None):
            raise Exception( 'Error: You are not allowed to specify both '+
                             '"factor" and "hidden" at the same time!' )
        
        if self.n_components is None:
            self.n_components = int(self.factor*dim_dummy)
        
        self.n_features = self.n_components
        
        ### start clock
        start_time = time.time()

        train_syn = anomaly_sampler( train, 
                                                 frac       = 0.5,
                                                 seed       = 1,
                                                 verbose    = 1 )
                        
            
        ### Combine real and synthetic datasets into one
        train_ref = pd.concat( [ train, train_syn] )
        train_ref.index = range(train_ref.shape[0])
        label_ref = pd.DataFrame( np.hstack( [ np.repeat('normal',train.shape[0]), 
                                 np.repeat('anomaly',train.shape[0]) ] ) )
        label_ref.index = range(label_ref.shape[0])
        label_ref.columns = ['label']
        label_ref['label'] = pd.Categorical(label_ref['label'])
            
        h2o.init(nthreads=6, max_mem_size=max_mem_size) # initialize H2O cluster
        data_ref = pd.concat([label_ref,train_ref], axis=1)
        train_ref_h2o = h2o.H2OFrame( data_ref ) # save train as H2OFrame
        train_ref_h2o[0]=train_ref_h2o[0].asfactor()
        
        if self.hidden is None:
            self.hidden = self.depth*[self.n_components]
        
        self.print_fit_started()
        
        ### Initialize the RF classifier which should learn to distinguish the 
        ### synthetic from the real data
        self.model = H2ODeepLearningEstimator( 
                                hidden          = self.hidden,
                                epochs          = self.epochs,
                                activation      = 'Tanh',
                                stopping_metric = 'AUC',
                                nfolds          = 5 )
        print self.model
        self.model.train( y=0, training_frame=train_ref_h2o )
        
        ### stop clock
        self.model_fit_finished(start_time)
        
        
    def compute_representation(self, data):
        data_h2o = h2o.H2OFrame( data )
        data_EE = self.model.deepfeatures(data_h2o, layer=self.rep_layer).as_data_frame()
        return data_EE;
    
###############################################################################


###############################################################################
### Sparse coding representation model ########################################
###############################################################################
"""
class SC_RepModel(RepModel):
    " ""
    DESCRIPTION
    " ""
    
    def __init__( self, 
                  factor       = None,
                  n_components = None,
                  hidden       = None, 
                  depth        = None,
                  rep_layer    = None, 
                  epochs       = None, 
                  ncores       = None, 
                  seed         = None,
                  verbose      = None  ):
        " ""
        DESCRIPTION
        " ""
        RepModel.__init__( self, 
                           ID = 'SC',
                           model_type = ( 'Sparse coding ' +
                                          'representation model' ), 
                           model_pars = [ 'factor','n_components'] )
    
        self.factor         = factor
        self.n_components   = n_components
        self.ncores         = ncores
        self.seed           = seed
        self.verbose        = verbose
        
        set_pars_by_config(self, config_Rep)
        
        
    def fit(self, train):
        
        ### Compute default n_components if not specified
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        dim_dummy = n_num_cols + np.sum(n_levels)
    
        if self.n_components is None:
            self.n_components = int(self.factor*dim_dummy)
        
        self.n_features = self.n_components
        
        ### start clock
        start_time = time.time()
        
########################################################################################################################
        
        ### ???
        self.model = ???
        
        ### stop clock
        self.model_fit_finished(start_time)
        
        
    def compute_representation(self, data):
        ???
        return data_EE;
"""
###############################################################################


###############################################################################
### Calculate dimensions of dataset ###########################################
###############################################################################

def get_dimensions(data):
    n_num_cols = 0
    n_cat_cols = 0
    n_levels = []
    for col in data:
        if data[col].dtype.name != 'category':
            n_num_cols += 1
        else:
            n_cat_cols += 1
            n_levels.append( len(np.unique(data[col])) )
    return n_num_cols, n_cat_cols, n_levels;

###############################################################################