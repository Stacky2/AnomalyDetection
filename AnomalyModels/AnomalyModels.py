# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 14:59:48 2017

@author: Mathias

= Anomaly Detection Models =
============================

This file contains the class definitions and functions of the following
Anomaly Detection Models:
    - Isolation Forest
    - Unsupervised Random Forest
    - Unsupervised XGB
    - K-means distance model
    - K-menas cluster size model
    - Autoencoder
    - Deep autoencoder
    - One-class SVM
    - Least-squares anomaly detection
    - Feature regressin and Classification model
    - Gaussian mixture model
    - Probabelistic PCA

"""


### Import modules
import pandas as pd
import numpy as np
import xgboost as xgb
import random as rd
import h2o
import time
import configparser
import ast
import os
import bisect
import math
import lsanomaly

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn import decomposition
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from random import sample
# from random import shuffle
from math import ceil
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn import mixture


### define max_mem_size allwoed for h2o (in GB, choose none for laptop)
max_mem_size = 16


### define config parser
config = configparser.ConfigParser()
config.read( os.path.join(os.path.dirname(__file__), "config.ini"))


###############################################################################
### Base class Anomaly Model ##################################################
###############################################################################

class AnomalyModel:
    """Base class for anomaly detction models"""
    def __init__(self, ID, model_type, model_pars, mode, verbose, ncores, 
                 seed):
        """Initialize an AnomalyModel
        
        Initializes an AnomalyModel base model that specifies variables and
        function that are shared by the different AnomalyModels.
        
        Parameters:
        -----------
        :param ID: 
            'ID' needs to be two to four character shortcut that specifies the 
            specific type of an AnomalyModel (eg. Isolation forest: 'IF',
            Feature regression and Classifictaion model: 'FRaC'). 
            This identification shortcut is important because it has to be
            the same at the beginning of the model 
            (eg. ID='IF': IF_AnomalyModel) and it is used as well to define 
            the default parameters in the "config.ini" file 
            (eg. ID="IF": [IF_Nov], [IF_Out] are the corresponding sections)
            
        :param model_type:
            'model_type' is a string that captures the name with which the 
            anomaly detection algorithm is addressed in console output or in
            the performance sheet.
            
        :parm model_pars:
            'model_pars' is a list of parameters that are specified in the 
            specific instance of an AnomalyModel and that determines the 
            reproducible (!) behaviour of an AnomalyModel.
        
        :param mode:
            'mode' is either "Novelty" or "Outlier" and determines the anomaly
            detection scenario for which the AnomalyModel should be used.
            
        :param verbose:
            'verbose' is either 0 or 1 and specifies if the algorithm should
            run silent or print console output respectively.
            
        :param ncores:
            'ncores' is an integer that specifies the maximal number of 
            threads that the algorithm is allowed to use."""
            
        self.model              = None
        self.model_type         = model_type
        self.ID                 = ID
        self.model_pars         = model_pars
        self.mode               = mode
        self.ncores             = ncores
        self.seed               = seed
        self.verbose            = verbose       # verbosity of output       
        
        self.enable_cdf_score   = False #############################################################################################################################

        self.performance        = None
        self.fit_time           = None          # elapsed time for fit
        self.score_time         = None          # elapsed time for scoring 
        
        
    def set_params(self, **params):
        """Sets all the variables of an instance according to the key-value 
        pairs given in the dictionary **params."""
        for par in params:
            setattr(self, par, params[par])
                                                
    ### printing & saving functions for get_anomaly_scores and get_performance
    def print_fit_started(self):
        """Print start of anomaly model fit and print values of model 
        parameters specified in model_types."""
        if self.verbose == 1:
            print ( '> Fit ' + self.model_type + ' anomaly model...' )
            for i in range(len(self.model_pars)):
                par = self.model_pars[i]
                print( '     ' + (par + 20*' ')[:20] 
                       + '= ' + repr(getattr(self, par)) )

    def model_fit_finished(self, start_time, train):
        """Print end of anomaly model fitting process and save elapsed time."""
        ##############################################################################################################
        if self.enable_cdf_score is True:
            self.train_scores = self.get_anomaly_scores(train, train_mode=True)
        ##############################################################################################################    
        
        time_elapsed = round(time.time() - start_time, 1)
        self.fit_time = time_elapsed
        if self.verbose == 1:
            print ('  model fit finished! (time elapsed: ' 
                    + repr(self.fit_time) + 's)\n') 
            
    def print_score_computation(self, data):
        """Print information about anomaly scores getting computed."""
        if self.verbose == 1:
            print ( '> Compute ' + self.ID + ' scores ('
                     + repr(data.shape[0]) + ' samples) ...' )
            
    def score_computation_finished(self, start_time):    
        """Print information about anomaly scores that have been computed."""
        time_elapsed = round( time.time() - start_time, 1 ) # stop clock
        self.score_time = time_elapsed
        if self.verbose == 1:
            print ( 'Score calculation finished! (time elapsed: ' 
                    + repr(self.score_time) + 's)\n'              )

    
    ### function to compute anomaly scores, core function 
    ### compute_anomaly_scores for actual computation needs to be specified
    ### in specific anomaly model
    def get_anomaly_scores( self, data, train_mode=False, cdf_score=False ):
        """A function to compute the anomaly scores
        
        The function that formally handles  the computation of anomaly scores
        of an anomaly model. The concrete computation given 'data' has still
        to be specified as a function of the specific AnomalyModel with the
        function compute_anomaly_scores(self, data).
        
        Parameters
        ----------
        :param data
            The DataFrame 'data' needs to have the same columns as the 
            DataFrame 'train' that has been used in the fit(self, train)
            function of an AnomalyModel."""
        
        ### start clock
        start_time = time.time()
    
        ### Print information of scores being computed
        self.print_score_computation(data)
        
        scores = self.compute_anomaly_scores(data)
        
        
        ######################################################################################################
        if (self.enable_cdf_score is True) & (train_mode is False) & (cdf_score is True):
            for i in range(len(scores)):
                scores[i] = np.mean(self.train_scores <= scores[i])
        ######################################################################################################

        ### Print that score calculation finished and time elapsed
        self.score_computation_finished(start_time)
        
        ### return the calculated scores
        return scores;
            
    def get_performance( self, test, label_test, print_rd_scores=-1 ):
        """A function that computes performance of an AnomalyModel
        
        This function computes the (AUC-) performance of an AnomalyModel that 
        has been trained with the fit(self, train) function. The function
        computes the anomaly scores for the data 'test' (which has to have
        the same columns as the set 'train') and computes the AUC performance
        of the scores compared with the correct labels given in label_test.
        
        Parameters:
        -----------
            :param test:
                The DataFrame 'test' needs to have the same columns as the 
                DataFrame 'train' that has been used in the fit(self, train)
                function of an AnomalyModel.
                
            :param label_test:
                An array of length test.shape[0] with values 0 or 1 resp. that
                specifies if the corresponding row in 'test' is a normal 
                instance or an anomaly resp.
                
            :param print_rd_scores:
                (default: -1) If an integer bigger 0 is given, the function
                prints out a number of 'print_rd_scores' anomaly scores
                computed for 'test' and a statistic the distribution of the
                scores.""" 
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if self.verbose == 1:
            print ( '\n=== TEST: ' + self.model_type 
                    + ' anomaly score ===' )
        
        ### get anomaly scores and AUC score
        scores = self.get_anomaly_scores( test )
        scores = np.maximum(np.minimum(scores, 1e10), -1e10)
        self.performance = roc_auc_score( label_test, scores )
        
        ### print random sample of scores if enabled
        if print_rd_scores > 0:
            print ( 'Random sample of ' + repr(print_rd_scores) +
                    ' anomaly scores\n' +
                    repr(sample(scores, print_rd_scores)) + '\n' +
                    'Summary of anomaly scores of test set:\n' +
                    repr(pd.DataFrame(scores).describe()) + '\n'   )
            
        ### print AUC score and time elapsed
        time_elapsed = round( time.time() - start_time, 1 ) # stop clock
        if self.verbose == 1:
            print ( '> RESULT: (time elapsed: ' 
                    + repr(time_elapsed) + 's)\n' +
                    '     AUC: ' + repr(self.performance) + '\n' +
                    '===========================================\n' )
            
        ### return AUC performance score
        return self.performance;

###############################################################################


###############################################################################
### get default parameters from config file ###################################
###############################################################################

def set_pars_by_config(self, config, mode):
    """Set parameters according to 'config'-configparser and 'mode' specifying
    the anomaly detection mode ("Novelty" or "Outlier")."""
    sec = self.ID + '_' + mode[:3]
    for key in config[sec]:
        if not hasattr(self, key) or (getattr(self, key) is None):
            setattr(self, key, ast.literal_eval(config[sec][key]))

###############################################################################


###############################################################################
### Isolation Forest anomaly model ############################################
###############################################################################

class IF_AnomalyModel(AnomalyModel):
    """Isolation forest anomaly model class"""
    
    def __init__( self,
                  mode           = None,
                  n_estimators   = None, 
                  sample_frac    = None, 
                  ncores         = None, 
                  seed           = None,
                  verbose        = None  ):
        """A function to initialize an isolation forest anomaly model
        
        Initializes the anomaly detection model with the parameters as
        specified in the function call or if not specified with the defaults 
        taken from the config file "config.ini" which has to be in the same 
        folder as this script.
        
        Parameters
        ----------
        :param mode:
            Either "Novelty" or "Outlier", specifies the default values used
            to fit the Isolation Forest, depending on wether it should used in
            used for a novelty detection / semi-supervised anomaly detection 
            or an outlier detection / unsupervised anomaly detection scenario.
            
        :param n_estimators:
            'n_estimators' is an integer that specifies number of iTrees 
            (isolation trees) that are fitted in the iForest (isolation 
            forest).
            
        :param sample_frac:
            'sample_frac' is a real number in (0,1] fraction of samples that 
            should be used in each iTree.
            
        :param seed:
            'seed' is real number that specifies the random initialization for 
            reproducibility.
            
        :parm verbose:
            'verbose' is either 0 or 1 and specifies if the algorithm should
            give console output or not.
            
            
        [   Returns [CHANGE!!!]
            -------
            :return: [arg returned], [Description of [arg returned]].   ]
        """
        
        AnomalyModel.__init__( self, 
                               ID           = 'IF',
                               model_type   = 'Isolation Forest',
                               model_pars   = [ 'n_estimators', 
                                                'sample_frac',
                                                'seed'],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        

        self.n_estimators   = n_estimators
        self.sample_frac    = sample_frac
        self.max_samples    = None
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
        
    def fit(self, train):
        """A function that fits the anomaly model
        
        Parameters:
        -----------
        :param train:
            'train' is a DataFrame that has columns of type numerical 
            ("int"/"float") or of type "category"."""
        ### start clock    
        start_time = time.time() 
        
        ### Print information of model being fitted
        self.print_fit_started()
        
        ### Transform subsmaple fraction into number of samples
        if self.sample_frac>=1:
            self.max_samples=train.shape[0]
        else:
            self.max_samples=self.sample_frac
        
        ### Initialization and Fit of Isolation Forest
        self.model = IsolationForest( 
                        n_estimators  = self.n_estimators, 
                        max_samples   = self.max_samples,
                        n_jobs        = self.ncores,
                        random_state  = self.seed, 
                        verbose       = self.verbose       )   
        
        
        train_num = lexical_enc_set(train)
        self.model.fit(train_num)
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)
        
        
    def compute_anomaly_scores(self, data):
        """A function that computes concrete anomaly scores."""
        # (revert scores (because output -1 means anomalous and 1 normal) 
        # and convert scores in [-1,1] to scores in [0,1])
        data_num = lexical_enc_set(data)
        scores = ( 1-self.model.decision_function(data_num) ) / 2
    
        return np.array(scores);

###############################################################################


###############################################################################
### Unsupervised Random Forest anomaly model ##################################
###############################################################################

class URF_AnomalyModel(AnomalyModel):
    """DESCRIPTION"""
    
    def __init__( self, 
                  mode                  = None,
                  n_estimators          = None, 
                  min_impurity_split    = None, 
                  #shuffeling            = None, 
                  frac                  = None,
                  enable_train_score    = None, 
                  ncores                = None, 
                  seed                  = None,
                  verbose               = None  ):
        """DESCRIPTION"""
        
        AnomalyModel.__init__( self, 
                               ID           = 'URF',
                               model_type   = 'Unsupervised Random Forest',
                               model_pars   = [ 'n_estimators', 
                                                'min_impurity_split',
                                                #'shuffeling',
                                                'frac',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        

        self.n_estimators       = n_estimators
        self.min_impurity_split = min_impurity_split
        #self.shuffeling         = shuffeling 
        self.frac               = frac
             
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)


    def fit(self, train):
        """ DESCRIPTION """

        ### start clock    
        start_time = time.time() 
        
        ### Print information of model being fitted
        self.print_fit_started()
        
        ### Initialize the RF classifier which should learn to distinguish the 
        ### synthetic from the real data
        self.model = RandomForestClassifier( 
                        n_estimators       = self.n_estimators,
                        min_impurity_split = self.min_impurity_split,
                        n_jobs             = self.ncores,
                        random_state       = self.seed,
                        verbose            = self.verbose )
   
        ### generate synthetic set with removed structure by independently
        ### permuting each column
        train_num = lexical_enc_set(train)
        
        #if ( self.shuffeling=='complete' ):
        #    if self.verbose == 1:
        #        print "  Create completely shuffeled anomaly dataset" 
        #        
        #    train_syn = train_num.copy()
        #    for i in range( train_syn.shape[1] ):
        #        train_syn.ix[:,i] = shuffle( train_syn.ix[:,i] ).values
                
        #elif ( self.shuffeling=='partial' ):
        if self.verbose == 1:
            print "  Create partially shuffeled anomaly dataset" 
                
        train_syn = anomaly_sampler( train_num, 
                                    frac       = self.frac,
                                    seed       = self.seed,
                                    verbose    = self.verbose )
    
        ### Combine real and synthetic datasets into one
        train_ref = pd.concat( [train_num, train_syn] )
        label_ref = np.hstack( [ np.repeat(0,train_num.shape[0]), 
                                 np.repeat(1,train_num.shape[0]) ] )
        
        self.model.fit( train_ref, label_ref )
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)
       
        
    def compute_anomaly_scores(self, data):
        data_num = lexical_enc_set(data)
        scores = self.model.predict_proba( data_num )[:,1]
    
        return np.array(scores);

###############################################################################


###############################################################################
### Unsupervised XGB anomaly model ############################################
###############################################################################

class UXGB_AnomalyModel(AnomalyModel):
    def __init__( self,
                  mode                  = None,
                  nrounds               = None, 
                  max_depth             = None, 
                  num_parallel_tree     = None, 
                  #shuffeling            = None, 
                  frac                  = None, 
                  enable_train_score    = None,
                  ncores                = None, 
                  seed                  = None,
                  verbose               = None  ):
        """DESCRIPTION"""
        
        AnomalyModel.__init__( self, 
                               ID           = 'UXGB',
                               model_type   = 'Unsupervised XGB',
                               model_pars   = [ 'nrounds',
                                                'max_depth',
                                                'num_parallel_tree',
                                                #'shuffeling',
                                                'frac',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        

        self.nrounds = nrounds
        self.max_depth = max_depth
        self.num_parallel_tree = num_parallel_tree
        #self.shuffeling = shuffeling
        self.frac = frac     

        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)


    def fit(self, train):
        ### start clock    
        start_time = time.time() 
        
        ### Print information of model being fitted
        self.print_fit_started()
        
        self.params = { 'eta':                0.01,
                        'gamma':              1,
                        'max_depth':          self.max_depth,
                        'min_child_weight':   1,
                        'subsample':          0.7,
                        'colsample_bytree':   0.7,
                        'alpha':              1,
                        'objective':          'binary:logistic',
                        'eval_metric':        'auc',
                        'num_parallel_tree':  self.num_parallel_tree,
                        'nthread':            self.ncores,
                        'verbose_eval':       True,
                        'seed':               self.seed }
   
        ### generate synthetic set with removed structure by independently
        ### permuting each column
        train_num = lexical_enc_set(train)
        """
        if ( self.shuffeling=='complete' ):
            if self.verbose == 1:
                print "  Create completely shuffeled anomaly dataset..." 
            train_syn = train_num.copy()
            for i in range(train_syn.shape[1]):
                train_syn.ix[:,i] = shuffle(train_syn.ix[:,i]).values
        """
        #elif ( self.shuffeling=='partial'):
        if self.verbose==1:
            print "  Create partially shuffeled anomaly dataset..." 
                
        train_syn = anomaly_sampler( train_num, 
                                     frac       = self.frac,
                                     seed       = self.seed,
                                     verbose    = self.verbose )
    
        train_ref = pd.concat([train_num, train_syn])
        label_ref = np.hstack( [ np.repeat(0,train_num.shape[0]), 
                                 np.repeat(1,train_num.shape[0]) ] )
        
        dtrain = xgb.DMatrix(train_ref, label=label_ref)
            
        self.model = xgb.train( self.params, dtrain, 
                                num_boost_round=self.nrounds)
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)
        
        
    def compute_anomaly_scores(self, data):

        data_num = lexical_enc_set(data)
        ddata = xgb.DMatrix( data_num )
        scores = self.model.predict( ddata )
    
        return np.array(scores);
        
###############################################################################


###############################################################################
### k-means dinstance anomaly model ###########################################
###############################################################################

class KMD_AnomalyModel(AnomalyModel):
    def __init__( self, 
                  mode              = None,
                  k                 = None,
                  n                 = None, 
                  max_iterations    = None, 
                  subsample         = None, 
                  colsample         = None,
                  MSE_normalization = None,
                  ncores            = None, 
                  seed              = None,
                  verbose           = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'KMD',
                               model_type   = 'k-means distance model',
                               model_pars   = [ 'k',
                                                'n',
                                                'max_iterations',
                                                'subsample',
                                                'colsample',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        

        self.k                  = k
        self.n                  = n
        self.max_iterations     = max_iterations                
        self.subsample          = subsample                
        self.colsample          = colsample                 
        self.MSE_normalization  = MSE_normalization 
      
        self.model              = []           
        self.dropped_cols       = []
        self.cat_cols           = []
        self.means              = None
        self.sds                = None

        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
    
            
    def fit(self, train):
        ### functions
        def get_mean_sd(data):
            cat_cols = []
            means = {}
            sds = {}
            for column in data:
                if ( data[column].dtype.name != 'category' ): 
                    cat_cols.append(column)
                    means[column] = np.mean(data[column])
                    sds[column] = np.std(data[column])
                    if (sds[column] == np.nan):
                        sds[column] = 1
                
            return cat_cols, means, sds;
       
        ### start clock    
        start_time = time.time() 
        
        ### Print information of model being fitted
        self.print_fit_started()

        if self.verbose != 1:
            h2o.no_progress()
            
        ### initialize H2O cluster     
        h2o.init(nthreads=self.ncores, max_mem_size=max_mem_size) 
       
        ### get mean and standard deviation of training data
        self.cat_cols, self.means, self.sds = get_mean_sd(train)
        train_H2O = h2o.H2OFrame(train) # save train as H2OFrame
                   

        ### fit an ensemble of n k-means cluster models
        k_save = max(1, min( int(train.shape[0]/10.0) , self.k) )
        for i in range(self.n):
           KMD_model = H2OKMeansEstimator( 
                          k                 = k_save, #self.k, 
                          max_iterations    = self.max_iterations, 
                          standardize       = True,
                          ignore_const_cols = False,
                          seed              = self.seed+i )
           
           if ( self.subsample>=1 ):
               subsample_H2O = train_H2O
           else:
               subsample_H2O = train_H2O.split_frame(ratios = [self.subsample], 
                                                     seed = self.seed+i)[0]
           
           if ( self.colsample>=1 ):
               cols_drop = []
               KMD_model.train(training_frame=subsample_H2O) # NEW!!
           else:
               cols_drop = sample(range(1, train.shape[1]+1), 
                                  int(train.shape[1]*(1-self.colsample)) ) # NEW !!
               KMD_model.train(training_frame=subsample_H2O.drop(cols_drop)) # NEW!!
               
           
           self.dropped_cols.append(cols_drop)
           
           # KMD_model.train(training_frame=subsample_H2O[0]) # NEW!!
           self.model.append(KMD_model)

           
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)
            
        if self.verbose != 1:
            h2o.show_progress()
        

    def compute_anomaly_scores(self, data):
            
        ### functions
        def standardize(data, cat_cols, means, sds):
            data_std = data.copy()
            for column in data:
                if column in cat_cols: 
                    data_std[column] = (data[column]-means[column]) / sds[column]
                
            return data_std;  
            
        def MSE( obs ):
            return np.mean((obs)**2);
        
        ### define model-independent datasets
        data_H2O = h2o.H2OFrame( data )
        data_std = standardize(data, self.cat_cols, self.means, self.sds)
        scores_list = pd.DataFrame( np.nan, 
                                   index   = range(data.shape[0]), 
                                   columns = range(self.n)              )
        
        for i in range(self.n):
            ### predict cluster-membership
            cl_idx = self.model[i].predict( data_H2O )
            cl_idx = cl_idx.as_data_frame() # convert H2O df to df
            
            ### names of the columns not dropped
            col_idx = [val for val in range(1, data.shape[1]+1) 
                          if val not in np.abs(self.dropped_cols[i]) ]
            col_idx_0 = [val-1 for val in col_idx]
            names = data.columns[ col_idx_0 ] ## WHY NEGATIVE???
            
            
            ### predict cluster-center of each observation
            cts = self.model[i].centers() # read out centers
            cts = pd.DataFrame.from_records(cts) # convert H2ODF to df
            cts.columns = names # assign right names
            cts_std = standardize(cts, self.cat_cols, self.means, self.sds)
            obs_cts_std = cts_std.ix[cl_idx["predict"]]
            obs_cts_std.columns = names # assign right names
            
            ### initialize distances with observations 
            dist = data_std.loc[:,names].copy() # initialize distance matrix
            
            ### compute distances 
            for column in obs_cts_std:
                # case: categorical
                if ( data[column].dtype.name == 'category' ):  
                    data_string = data_std[column].values.astype('string') 
                    cluster_string = obs_cts_std[column].values.astype(
                                                                   'string')
                    dist[column] = (data_string != cluster_string).astype(
                                                                      'int')
                # case: continuous
                else: 
                    dist[column] = ( data_std[column].values
                    - obs_cts_std[column].values )
        
                
            distances = dist.apply(MSE, axis=1)
            
            cluster_sizes = np.array(self.model[i].size())
            cluster_MSEs = np.array(self.model[i].withinss()) / cluster_sizes
            
            #obs_cluster_sizes = cluster_sizes[cl_idx["predict"]]
            obs_MSEs = cluster_MSEs[cl_idx["predict"]]
            
            
            if self.MSE_normalization:
                scores = 1 - np.exp( - distances/obs_MSEs )
            else:
                scores = 1 - np.exp( - distances )
                
            # scores = 1 - np.exp( - (distances/obs_cluster_sizes) )
        
            scores_list[i] = scores.values
        
   
        ### aggregate the scores of different models
        if self.mode == 'Novelty':
            scores_final = scores_list.apply(np.mean, axis=1)
        
        if self.mode == 'Outlier':
            scores_final = scores_list.apply(min, axis=1)

        return np.array(scores_final);

        #return scores_list;
        
    """
    def get_performance(self, test, label_test, print_rd_scores=-1):
         
        start_time = time.time() # start clock
        
        if self.verbose != 1:
            h2o.no_progress()
        
        if self.verbose == 1:
            print '\n=== TEST: k-means distance anomaly score ==='
        
            ### define function
        def RMS(x):
            p = -2
            # score = (x^p)^(1/p)
            score = np.sum(x**p)**(1/p)
            return score;
        
        
        ### get anomaly scores and AUC score
        scores = self.get_anomaly_scores( test )
        
        ### print random sample of scores if enabled
        if self.verbose == 1:
            if print_rd_scores > 0:
                print ( 'Random sample of ' + repr(print_rd_scores) +
                        ' anomaly scores\n' +
                        repr(sample(scores[0], print_rd_scores)) + '\n' +
                        'Summary of anomaly scores of test set:\n' +
                        repr(pd.DataFrame(scores).describe()) + '\n'   )
            
        
        time_elapsed = round(time.time() - start_time, 1) # stop clock
            
        ### print AUC score
        if self.verbose == 1:
            print ( '> RESULT: (time elapsed: ' 
                    + repr(time_elapsed) + 's)\n')
        
        if self.verbose == 1:
            print 'Scores of individual clusterings:'
            for i in range(scores.shape[1]):
                auc_score = roc_auc_score( label_test, scores.loc[:,i] )
                print 'Model ' + repr(i) + ': ' + repr(auc_score)
        
        ### print AUC scores of combined model scores
        if self.verbose == 1:
            print '\nScores of combined models:'
            
        if self.verbose == 1:
            min_scores = scores.apply(min, axis=1)
            min_auc_score = roc_auc_score( label_test, min_scores )
            print 'min-score:    ' + repr(min_auc_score)
            
        if self.verbose == 1:
            max_scores = scores.apply(max, axis=1)
            max_auc_score = roc_auc_score( label_test, max_scores )
            print 'max-score:    ' + repr(max_auc_score)
            
        mean_scores = scores.apply(np.mean, axis=1)
        mean_auc_score = roc_auc_score( label_test, mean_scores )
        if self.verbose == 1:
            print 'mean-score:   ' + repr(mean_auc_score)
            
        if self.verbose == 1:
            median_scores = scores.apply(np.median, axis=1)
            median_auc_score = roc_auc_score( label_test, median_scores )
            print 'median-score: ' + repr(median_auc_score)
            
        if self.verbose == 1:
            rms_scores = scores.apply( RMS, axis=1 )
            rms_auc_score = roc_auc_score( label_test, rms_scores )
            print 'RMS-score:    ' + repr(rms_auc_score)
            
        if self.verbose == 1:
            print '======================================================\n'
    
        if self.verbose != 1:
            h2o.show_progress()    
        
        self.performance = mean_auc_score
        return self.performance;
    """
###############################################################################


###############################################################################
### k-means cluster size anomaly model ########################################
###############################################################################

class KMC_AnomalyModel(AnomalyModel):
    def __init__( self, 
                  mode              = None,
                  k                 = None,
                  n                 = None, 
                  max_iterations    = None, 
                  subsample         = None, 
                  colsample         = None,
                  MSE_normalization = None,
                  ncores            = None, 
                  seed              = None,
                  verbose           = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'KMC',
                               model_type   = 'k-means cluster size model',
                               model_pars   = [ 'k',
                                                'n',
                                                'max_iterations',
                                                'subsample',
                                                'colsample',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        

        self.k                  = k
        self.n                  = n
        self.max_iterations     = max_iterations                
        self.subsample          = subsample                
        self.colsample          = colsample                 
        self.MSE_normalization  = MSE_normalization 
      
        self.model              = []           
        self.dropped_cols       = []
        self.cat_cols           = []
        self.means              = None
        self.sds                = None

        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
    
            
    def fit(self, train):
        ### functions
        def get_mean_sd(data):
            cat_cols = []
            means = {}
            sds = {}
            for column in data:
                if ( data[column].dtype.name != 'category' ): 
                    cat_cols.append(column)
                    means[column] = np.mean(data[column])
                    sds[column] = np.std(data[column])
                    if (sds[column] == np.nan):
                        sds[column] = 1
                
            return cat_cols, means, sds;
       
        ### start clock    
        start_time = time.time() 
        
        ### Print information of model being fitted
        self.print_fit_started()

        if self.verbose != 1:
            h2o.no_progress()
            
        ### initialize H2O cluster     
        h2o.init(nthreads=self.ncores, max_mem_size=max_mem_size) 
       
        ### get mean and standard deviation of training data
        self.cat_cols, self.means, self.sds = get_mean_sd(train)
        train_H2O = h2o.H2OFrame(train) # save train as H2OFrame
                   

        ### fit an ensemble of n k-means cluster models
        k_save = max(1, min( int(train.shape[0]/10.0) , self.k) )
        for i in range(self.n):
           KMC_model = H2OKMeansEstimator( 
                          k                 = k_save, #self.k, 
                          max_iterations    = self.max_iterations, 
                          standardize       = True,
                          ignore_const_cols = False,
                          seed              = self.seed+i )
           
           if ( self.subsample>=1 ):
               subsample_H2O = train_H2O
           else:
               subsample_H2O = train_H2O.split_frame(ratios = [self.subsample], 
                                                     seed = self.seed+i)[0]
           
           if ( self.colsample>=1 ):
               cols_drop = []
               KMC_model.train(training_frame=subsample_H2O) # NEW!!
           else:
               cols_drop = sample(range(1, train.shape[1]+1), 
                                  int(train.shape[1]*(1-self.colsample)) ) # NEW !!
               KMC_model.train(training_frame=subsample_H2O.drop(cols_drop)) # NEW!!
               
           
           self.dropped_cols.append(cols_drop)
           
           # KMC_model.train(training_frame=subsample_H2O[0]) # NEW!!
           self.model.append(KMC_model)

           
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)
            
        if self.verbose != 1:
            h2o.show_progress()
        

    def compute_anomaly_scores(self, data):
            
        ### functions
        def standardize(data, cat_cols, means, sds):
            data_std = data.copy()
            for column in data:
                if column in cat_cols: 
                    data_std[column] = (data[column]-means[column]) / sds[column]
                
            return data_std;  
            
        def MSE( obs ):
            return np.mean((obs)**2);
        
        ### define model-independent datasets
        data_H2O = h2o.H2OFrame( data )
        data_std = standardize(data, self.cat_cols, self.means, self.sds)
        scores_list = pd.DataFrame( np.nan, 
                                   index   = range(data.shape[0]), 
                                   columns = range(self.n)              )
        
        for i in range(self.n):
            ### predict cluster-membership
            cl_idx = self.model[i].predict( data_H2O )
            cl_idx = cl_idx.as_data_frame() # convert H2O df to df
            
            ### names of the columns not dropped
            col_idx = [val for val in range(1, data.shape[1]+1) 
                          if val not in np.abs(self.dropped_cols[i]) ]
            col_idx_0 = [val-1 for val in col_idx]
            names = data.columns[ col_idx_0 ] ## WHY NEGATIVE???
            
            
            ### predict cluster-center of each observation
            cts = self.model[i].centers() # read out centers
            cts = pd.DataFrame.from_records(cts) # convert H2ODF to df
            cts.columns = names # assign right names
            cts_std = standardize(cts, self.cat_cols, self.means, self.sds)
            obs_cts_std = cts_std.ix[cl_idx["predict"]]
            obs_cts_std.columns = names # assign right names
            
            ### initialize distances with observations 
            dist = data_std.loc[:,names].copy() # initialize distance matrix
            
            ### compute distances 
            for column in obs_cts_std:
                # case: categorical
                if ( data[column].dtype.name == 'category' ):  
                    data_string = data_std[column].values.astype('string') 
                    cluster_string = obs_cts_std[column].values.astype(
                                                                   'string')
                    dist[column] = (data_string != cluster_string).astype(
                                                                      'int')
                # case: continuous
                else: 
                    dist[column] = ( data_std[column].values
                    - obs_cts_std[column].values )
        
                
            # distances = dist.apply(MSE, axis=1)
            
            cluster_sizes = np.array(self.model[i].size())
            cluster_MSEs = np.array(self.model[i].withinss()) / cluster_sizes
            
            obs_cluster_sizes = cluster_sizes[cl_idx["predict"]]
            obs_MSEs = cluster_MSEs[cl_idx["predict"]]
            
            
            if self.MSE_normalization:
                scores = 1 - np.exp( - 1/(obs_cluster_sizes*obs_MSEs) )
            else:
                scores = 1 - np.exp( - 1/obs_cluster_sizes )
                
            # scores = 1 - np.exp( - (distances/obs_cluster_sizes) )
        
            scores_list[i] = scores
        
        ### aggregate the scores of different models
        if self.mode == 'Novelty':
            scores_final = scores_list.apply(np.mean, axis=1)
        
        if self.mode == 'Outlier':
            scores_final = scores_list.apply(min, axis=1)
                
        return np.array(scores_final);

        
        #return scores_list;
        
    """
    def get_performance(self, test, label_test, print_rd_scores=-1):
         
        start_time = time.time() # start clock
        
        if self.verbose != 1:
            h2o.no_progress()
        
        if self.verbose == 1:
            print '\n=== TEST: k-means distance anomaly score ==='
        
            ### define function
        def RMS(x):
            p = -2
            # score = (x^p)^(1/p)
            score = np.sum(x**p)**(1/p)
            return score;
        
        
        ### get anomaly scores and AUC score
        scores = self.get_anomaly_scores( test )
        
        ### print random sample of scores if enabled
        if self.verbose == 1:
            if print_rd_scores > 0:
                print ( 'Random sample of ' + repr(print_rd_scores) +
                        ' anomaly scores\n' +
                        repr(sample(scores[0], print_rd_scores)) + '\n' +
                        'Summary of anomaly scores of test set:\n' +
                        repr(pd.DataFrame(scores).describe()) + '\n'   )
            
        
        time_elapsed = round(time.time() - start_time, 1) # stop clock
            
        ### print AUC score
        if self.verbose == 1:
            print ( '> RESULT: (time elapsed: ' 
                    + repr(time_elapsed) + 's)\n')
        
        if self.verbose == 1:
            print 'Scores of individual clusterings:'
            for i in range(scores.shape[1]):
                auc_score = roc_auc_score( label_test, scores.loc[:,i] )
                print 'Model ' + repr(i) + ': ' + repr(auc_score)
        
        ### print AUC scores of combined model scores
        if self.verbose == 1:
            print '\nScores of combined models:'
            
        if self.verbose == 1:
            min_scores = scores.apply(min, axis=1)
            min_auc_score = roc_auc_score( label_test, min_scores )
            print 'min-score:    ' + repr(min_auc_score)
            
        if self.verbose == 1:
            max_scores = scores.apply(max, axis=1)
            max_auc_score = roc_auc_score( label_test, max_scores )
            print 'max-score:    ' + repr(max_auc_score)
            
        mean_scores = scores.apply(np.mean, axis=1)
        mean_auc_score = roc_auc_score( label_test, mean_scores )
        if self.verbose == 1:
            print 'mean-score:   ' + repr(mean_auc_score)
            
        if self.verbose == 1:
            median_scores = scores.apply(np.median, axis=1)
            median_auc_score = roc_auc_score( label_test, median_scores )
            print 'median-score: ' + repr(median_auc_score)
            
        if self.verbose == 1:
            rms_scores = scores.apply( RMS, axis=1 )
            rms_auc_score = roc_auc_score( label_test, rms_scores )
            print 'RMS-score:    ' + repr(rms_auc_score)
            
        if self.verbose == 1:
            print '======================================================\n'
    
        if self.verbose != 1:
            h2o.show_progress()    
        
        self.performance = mean_auc_score
        return self.performance;
    """
###############################################################################


###############################################################################
### Autoencoder anomaly model #################################################
###############################################################################

class AE_AnomalyModel(AnomalyModel):
    def __init__( self,
                  mode                  = None,
                  hidden                = None,
                  factor                = None,
                  epochs                = None,
                  l1                    = None,
                  enable_train_score    = None,
                  ncores                = None,
                  seed                  = None,
                  verbose               = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'AE',
                               model_type   = 'Autoencoder model',
                               model_pars   = [ 'hidden',
                                                'epochs',
                                                'l1',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.hidden = hidden
        self.factor = factor
        self.epochs = epochs
        self.l1 = l1

        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
            
    def fit(self, train):
        start_time = time.time() # start clock
        
        if self.verbose != 1:
            h2o.no_progress()
        
        ### get dimensions of numerical and categorical features
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        size_input = n_num_cols + np.sum(n_levels)
        
        if self.hidden is None:
            self.hidden = [int(size_input*self.factor)]
        
        self.print_fit_started()
        
        h2o.init(nthreads=self.ncores, max_mem_size=max_mem_size) # initialize H2O cluster
        train_h2o = h2o.H2OFrame(train) # save train as H2OFrame

        
        self.model = H2OAutoEncoderEstimator(
                        activation      = "Tanh",
                        hidden          = self.hidden,
                        epochs          = self.epochs,
                        l1              = self.l1,
                        score_interval  = 1,
                        stopping_rounds = 0,
                        overwrite_with_best_model = True,
                        seed = self.seed )
        self.model.train(training_frame=train_h2o) 
      
        self.model_fit_finished(start_time, train)
            
        if self.verbose != 1:
            h2o.show_progress()
 
        
    def compute_anomaly_scores(self, data):

        data_h2o = h2o.H2OFrame(data)
        AE_score_h2o = self.model.anomaly( data_h2o )
        RecMSE = AE_score_h2o.as_data_frame()['Reconstruction.MSE']
        scores = 1/(1+np.exp(-RecMSE))
        return np.array(scores);
        
###############################################################################


############################################################################### 
### Deep Autoencoder anomaly model ############################################
############################################################################### 

class DAE_AnomalyModel(AnomalyModel):
    def __init__( self,
                  mode                  = None,
                  hidden                = None,
                  factor                = None,
                  epochs                = None,
                  l1                    = None,
                  enable_train_score    = None,
                  ncores                = None,
                  seed                  = None,
                  verbose               = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'DAE',
                               model_type   = 'Deep autoencoder model',
                               model_pars   = [ 'hidden',
                                                'epochs',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.hidden = hidden   
        self.factor = factor
        self.epochs = epochs
        self.l1     = l1
        
        self.model  = []
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
    def fit(self, train):
        ### functions
        def mirror(layer):
            layer_rev = layer[:]
            layer_rev.reverse()
            hidden = layer + layer_rev[1:len(layer_rev)]
            return hidden;
            
    
        start_time = time.time() # start clock
        
        if self.verbose != 1:
            h2o.no_progress()
            
        ### get dimensions of numerical and categorical features
        n_num_cols, n_cat_cols, n_levels = get_dimensions(train)
        size_input = n_num_cols + np.sum(n_levels)
        
        if self.hidden is None:
            self.hidden = [ [ int(size_input*((self.factor+1)/2)) ], 
                            [ int(size_input*self.factor) ] ]


        self.print_fit_started()
        
        h2o.init(nthreads=self.ncores, max_mem_size=max_mem_size) # initialize H2O cluster
        train_h2o = h2o.H2OFrame(train) # save train as H2OFrame


        for layer in self.hidden:
                
            layer_final = mirror(layer)
                
            if self.verbose == 1:
                print "Training " + repr(layer_final) + " - AE..."
                
            model = H2OAutoEncoderEstimator(
                                            activation      = "Tanh",
                                            hidden          = layer_final,
                                            epochs          = self.epochs,
                                            l1              = self.l1,
                                            score_interval  = 1,
                                            stopping_rounds = 0,
                                            overwrite_with_best_model = True,
                                            seed = self.seed
                                            )
            model.train(training_frame=train_h2o) 
            self.model.append(model)
                
            train_h2o = model.deepfeatures(train_h2o, math.ceil(len(layer)/2))
      
            
        self.model_fit_finished(start_time, train)
            
        if self.verbose != 1:
            h2o.show_progress()
        
        
    def compute_anomaly_scores(self, data):

        data_h2o = h2o.H2OFrame(data)
        for i in range(len(self.hidden)-1):
            data_h2o = self.model[i].deepfeatures(data_h2o, len(self.hidden[i])-1 )
        
        AE_score_h2o = self.model[len(self.hidden)-1].anomaly( data_h2o )
        RecMSE = AE_score_h2o.as_data_frame()['Reconstruction.MSE']
        scores = 1/(1+np.exp(-RecMSE))
        
        return np.array(scores);

############################################################################### 


############################################################################### 
### One-class SVM model #######################################################
###############################################################################

class OSVM_AnomalyModel(AnomalyModel):
    """DESCRIPTION"""
    
    def __init__( self,
                  mode           = None,
                  nu             = None, 
                  gamma          = None, 
                  max_samples    = None,
                  ncores         = None, 
                  seed           = None,
                  verbose        = None  ):

        AnomalyModel.__init__( self, 
                               ID           = 'OSVM',
                               model_type   = 'One-class SVM',
                               model_pars   = [ 'nu',
                                                'gamma',
                                                'max_samples',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.nu             = nu
        self.gamma          = gamma
        self.max_samples    = max_samples
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode) 
        
        
    def fit(self, train):
        start_time = time.time() # start clock
                    
        ### Print information of model being fitted
        self.print_fit_started()
        
        if self.gamma <= 0:
            self.gamma = 'auto'
        """
        ### Initialization and Fit of Isolation Forest
        self.model = svm.OneClassSVM( 
                        kernel        = 'rbf', 
                        nu            = self.nu,
                        gamma         = 'auto', #self.gamma,
                        random_state  = self.seed, 
                        cache_size    = 1000,
                        verbose       = self.verbose       )   
        """
        ### Initialization and Fit of Isolation Forest
        self.model = svm.OneClassSVM( 
                        kernel        = 'rbf', 
                        nu            = self.nu,
                        gamma         = 'auto', #self.gamma,
                        random_state  = self.seed, 
                        cache_size    = 1000,
                        verbose       = self.verbose       )   
        train_num = pd.get_dummies(train)
        
        ### if enabled, reduce the number of samples, to train the One-class
        ### SVM, to the size of max_samples
        if (self.max_samples > 0) and (self.max_samples < train.shape[0]):
            rd.seed(self.seed)
            IDX = range(train_num.shape[0])
            rd.shuffle(IDX)
            train_num_sel = train_num.loc[IDX[:self.max_samples],:]
        else:
            train_num_sel = train_num
            
        self.model.fit(train_num_sel)
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)

        
    def compute_anomaly_scores(self, data):

        # (revert scores (because output -1 means anomalous and 1 normal) 
        # and convert scores in [-1,1] to scores in [0,1])
        data_num = pd.get_dummies(data)
        scores = -self.model.decision_function(data_num)
        scores_flat = [item for sublist in scores for item in sublist]
        
        return np.array(scores_flat);

 
###############################################################################


############################################################################### 
### Least Squares anomaly detection model #####################################
###############################################################################

class LSAD_AnomalyModel(AnomalyModel):
    """DESCRIPTION"""
    
    def __init__( self,
                  mode           = None,
                  sigma          = None, 
                  rho            = None, 
                  max_samples    = None,
                  ncores         = None, 
                  seed           = None,
                  verbose        = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'LSAD',
                               model_type   = 'Least Squares anomaly detection',
                               model_pars   = [ 'sigma',
                                                'rho',
                                                'max_samples',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.sigma          = sigma
        self.rho            = rho
        self.max_samples    = max_samples
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
        
    def fit(self, train):
        
        start_time = time.time() # start clock
        
        ### Initialization and Fit of Isolation Forest
        print self.max_samples
        self.model = lsanomaly.LSAnomaly(rho=self.rho, sigma=self.sigma, n_kernels_max=self.max_samples) #sigma=3, rho=0.1)
            
        ### Print information of model being fitted
        self.print_fit_started()
        
        train_num = np.array(pd.get_dummies(train))
        
        ### if enabled, reduce the number of samples, to train the One-class
        ### SVM, to the size of max_samples
        self.model.fit(train_num)
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)

        
    def compute_anomaly_scores(self, data):

        data_num = np.array(pd.get_dummies(data))
        scores = pd.DataFrame(self.model.predict_proba(data_num))[1]

        return np.array(scores);

###############################################################################


############################################################################### 
### FRaC anomaly model ########################################################
############################################################################### 

class FRaC_AnomalyModel(AnomalyModel):
    
    def __init__( self,
                  mode                  = None, 
                  n_estimators          = None, 
                  bw                    = None,
                  ncores                = None, 
                  seed                  = None,
                  verbose               = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'FRaC',
                               model_type   = 'Feature Regression and Classification',
                               model_pars   = [ 'n_estimators',
                                                'bw',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        
        self.n_estimators           = n_estimators
        
        self.model                  = {}
        self.cols_sel               = None
        self.hists                  = {}
        self.probs                  = {}
        self.sm_probs               = {}
        self.entropy                = {}
        
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
        
    def fit(self, train):
            
        ### functions
        def drop_label_col_and_encode(train, column):
            train_temp = train.copy()
            train_temp.drop(column, axis=1, inplace=True)
            train_temp = lexical_enc_set(train_temp)
            return train_temp;
            
        def fit_RFClassifier(train, label):
            model = RandomForestClassifier( 
                       n_estimators       = self.n_estimators,
                       min_impurity_split = 1e-7,
                       n_jobs             = self.ncores,
                       random_state       = self.seed,
                       verbose            = 0 )
            model.fit(train_temp, label)
            return model;
        
        def fit_RFRegressor(train, label):
            model = RandomForestRegressor( 
                       n_estimators       = self.n_estimators,
                       min_impurity_split = 1e-7,
                       #max_depth  = 15,
                       n_jobs             = self.ncores,
                       random_state       = self.seed,
                       verbose            = 0 )
            model.fit(train_temp, label)
            return model;
            
        def get_probs_from_table(probs_table, label):
            probs_list = np.concatenate( probs_table )
            real_idx = [2*i for i in range(len(label))]+pd.factorize(label)[0]
            obs_probs = probs_list[real_idx]
            return obs_probs;
            
        def get_prob_from_diff_and_hist(diff, hist, sm_probs):
            breaks = hist[1]
            def get_bin_idx(value):
                return bisect.bisect_left(breaks, value);
            get_bin_idxs=np.vectorize(get_bin_idx)
            bin_idxs = get_bin_idxs(diff)
            obs_probs = sm_probs[bin_idxs]
            return obs_probs;
            
        def entropy(p):
            eps = 1e-8
            p = np.maximum(p, eps)
            return np.sum(-p*np.log(p));
            
            
        ### start clock
        start_time = time.time()
            
        print ( '> Fit FRaC model...\n' +
                '     number of trees = ' + repr(self.n_estimators) + '\n')
        
        ### remove all features with too few variance (because disturbes algo)
        self.cols_sel = list(train.columns[ lexical_enc_set(train).apply(np.var, axis=0) >= 1e-4 ])
        print self.cols_sel

        ### remove all categorical features whith levels that have lass then 10
        ### members (in the training set), because RF can't deal with them
        for col in train:
            if ( train[col].dtype.name == 'category' ): 
                if ( (min(train[col].value_counts())<10) & (col in self.cols_sel) ):
                    self.cols_sel.remove(col)
        
        for column in self.cols_sel:
            if ( train[column].dtype.name == 'category' ): 
                print 'Fit model for ' + column + ' (categorical)...'
                
                train_temp = drop_label_col_and_encode(train, column)
                
                model = fit_RFClassifier(train_temp, train[column])
                pred_probs_table = cross_val_predict(model, train_temp, train[column], 
                                              cv=5, method='predict_proba')
                cv_probs = get_probs_from_table(pred_probs_table, train[column])
                                
                self.model[column]=model
                self.entropy[column] = entropy(cv_probs)
                
            else:
                print 'Fit model for ' + column + ' (continuous)...'
                
                def GaussWeights(center, npoints, bw):
                    weights = [np.exp( - ( pt - center  )**2 / bw ) for pt in range(1,(npoints+1)) ]
                    normalized_weights = [wght/sum(weights) for wght in weights]
                    return normalized_weights;
                    
                def SmoothHistogram(counts, bw):
                    smoothed_counts = np.repeat(np.nan, len(counts))
                    for i in range(len(counts)):
                        smoothed_counts[i] = sum(GaussWeights(i+1, len(counts), bw=bw) * counts)
                    return (smoothed_counts/np.sum(smoothed_counts));

                def get_hists(predicted, label, bw):
                    hist = np.histogram( predicted - label, bins=int(np.sqrt(len(predicted)))) 
                    counts = hist[0]
                    freqs = counts/float(np.sum(counts))
                    probs = np.concatenate([[0],freqs,[0]])
                    sm_probs = SmoothHistogram(probs, bw=bw)
                    """ ## print the (smoothed and original) marignal histogram
                    plt.bar(range(len(sm_probs)), sm_probs)
                    plt.show()
                    plt.bar(range(len(probs)), probs)
                    plt.show()
                    """                    
                    return hist, probs, sm_probs;
                    
                    
                train_temp = drop_label_col_and_encode(train, column)
                
                model = fit_RFRegressor(train_temp, train[column])
                
                predicted = cross_val_predict(model, train_temp, train[column], 
                                              cv=5)

                hist, probs, sm_probs = get_hists(predicted, train[column], bw=self.bw)
                
                diff = predicted-train[column]
                cv_probs = get_prob_from_diff_and_hist(diff, hist, sm_probs)
                
                self.probs[column]=probs
                self.sm_probs[column]=sm_probs
                self.hists[column]=hist
                self.model[column]=model
                self.entropy[column] = entropy(cv_probs)
                
                
        self.model_fit_finished(start_time, train)


    def compute_anomaly_scores(self, data):
        
        ### functions
        def drop_label_col_and_encode(train, column):
            train_temp = train.copy()
            train_temp.drop(column, axis=1, inplace=True)
            train_temp = lexical_enc_set(train_temp)
            return train_temp;
            
        def get_probs_from_table(probs_table, label):
            probs_list = np.concatenate( probs_table )
            real_idx = [2*i for i in range(len(label))]+pd.factorize(label)[0]
            obs_probs = probs_list[real_idx]
            return obs_probs;
            
        def get_prob_from_diff_and_hist(diff, hist, sm_probs):
            breaks = hist[1]
            def get_bin_idx(value):
                return bisect.bisect_left(breaks, value);
            get_bin_idxs=np.vectorize(get_bin_idx)
            bin_idxs = get_bin_idxs(diff)
            obs_probs = sm_probs[bin_idxs]
            return obs_probs;
        
        start_time = time.time() # start clock
        
        scores_list = pd.DataFrame( np.nan, 
                                   index   = range(data.shape[0]), 
                                   columns = self.cols_sel         )
        
        for column in self.cols_sel:
            print 'Compute surprisal score for ' + column
            if ( data[column].dtype.name == 'category' ): 
                
                data_temp = drop_label_col_and_encode(data, column)
                
                probs_table=self.model[column].predict_proba(data_temp)
                
                obs_probs = get_probs_from_table(probs_table, data[column])

                eps = 1e-4
                scores_list[column] = -np.log( np.maximum(obs_probs,eps) ) - self.entropy[column]
                
            else:
                data_temp = drop_label_col_and_encode(data, column)
                
                diff = self.model[column].predict(data_temp) - data[column]
                
                sm_probs = self.probs[column]
                hist = self.hists[column]
                
                obs_probs = get_prob_from_diff_and_hist(diff, hist, sm_probs)
                
                eps = 1e-4
                scores_list[column] = -np.log( np.maximum(obs_probs,eps) ) - self.entropy[column]
        
        scores = scores_list.apply(np.mean, axis=1)

        return np.array(scores);
        #return scores_list;        
    """
    def get_performance(self, test, label_test, print_rd_scores=-1):
         
        start_time = time.time() # start clock
        
        print '\n=== TEST: FRaC anomaly score ==='
        
        ### get anomaly scores and AUC score
        surprisals = self.get_anomaly_scores( test )
        
        for col in surprisals:
            print col + ': ' + repr(roc_auc_score(label_test, surprisals[col]))
        
        scores = surprisals.apply(np.mean, axis=1)
        self.performance = roc_auc_score(label_test, scores)
        
        ### print random sample of scores if enabled
        if print_rd_scores > 0:
            print ( 'Random sample of ' + repr(print_rd_scores) +
                    ' anomaly scores\n' +
                    repr(sample(scores, print_rd_scores)) + '\n' +
                    'Summary of anomaly scores of test set:\n' +
                    repr(pd.DataFrame(scores).describe()) + '\n'   )
            
        
        time_elapsed = round(time.time() - start_time, 1) # stop clock
            
        ### print AUC score
        print ( '> RESULT: (time elapsed: ' 
                + repr(time_elapsed) + 's)\n' +
                '     AUC: ' + repr(self.performance) + '\n' +
                '======================================================\n' )
        return self.performance;
    """
###############################################################################


############################################################################### 
### Least Squares anomaly detection model #####################################
###############################################################################

class GMM_AnomalyModel(AnomalyModel):
    """DESCRIPTION"""
    
    def __init__( self,
                  mode           = None,
                  n_components   = None, 
                  ncores         = None, 
                  seed           = None,
                  verbose        = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'GMM',
                               model_type   = 'Gaussian mixture model',
                               model_pars   = [ 'n_components',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.n_components = n_components
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
        
    def fit(self, train):
        
        start_time = time.time() # start clock
            
        
        ### Print information of model being fitted
        self.print_fit_started()
        
        ### Initialization of GMM
        n_components_save = max(1, min( int(train.shape[0]/10) , self.n_components) ) 
        print n_components_save
        self.model = mixture.GaussianMixture(n_components=n_components_save )
        
        self.model.fit(pd.get_dummies(train))
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)

        
    def compute_anomaly_scores(self, data):

        scores = -self.model.score_samples(pd.get_dummies(data))

        return np.array(scores);

###############################################################################


############################################################################### 
### Least Squares anomaly detection model #####################################
###############################################################################

class PCA_AnomalyModel(AnomalyModel):
    """DESCRIPTION"""
    
    def __init__( self,
                  mode           = None,
                  decay_rate     = None, 
                  ncores         = None, 
                  seed           = None,
                  verbose        = None  ):
        
        AnomalyModel.__init__( self, 
                               ID           = 'PCA',
                               model_type   = 'PCA model',
                               model_pars   = [ 'decay_rate',
                                                'seed' ],
                               mode         = mode,
                               verbose      = verbose,
                               ncores       = ncores,
                               seed         = seed )
        
        self.decay_rate = decay_rate
        self.means = None
        self.sds = None
        
        if (self.mode != 'Novelty') & (self.mode != 'Outlier'):
            raise Exception('Choose either mode = "Novelty" or "Outlier".')
            
        ### Choose defaults according to the mode in which the method should be 
        ### used. If arguments are given take these else take defaults 
        ### that are specified for the mode in the AnomalyModels config file.
        set_pars_by_config(self, config, mode)
        
        
    def fit(self, train):
        
        def get_mean_sd(data):
            means = {}
            sds = {}
            for column in data:
                means[column] = np.mean(data[column])
                sds[column] = np.std(data[column])
                if (sds[column] == np.nan) | (sds[column] == 0):
                    sds[column] = 1

            return means ,sds;
        
        
        def standardize(data, means, sds):
            data_std = data.copy()
            for column in data:
                data_std[column] = (data[column]-means[column]) / sds[column]
                        
            return data_std;  
        
        
        start_time = time.time() # start clock
            
        
        ### Print information of model being fitted
        self.print_fit_started()
        
        ### Initialization of GMM
        train_dummy = pd.get_dummies(train)
        self.means, self.sds = get_mean_sd(train_dummy)
        train_dummy = standardize(train_dummy, self.means, self.sds)
        
        #self.cols_dropped = list(train_dummy.columns[ train_dummy.apply(np.var, axis=0) < 1e-4 ])
        #print self.cols_dropped
        
        #train_dummy = train_dummy.drop(self.cols_dropped, axis=1)
        
        self.n_components = train_dummy.shape[1]-1
        self.model = decomposition.PCA( n_components = self.n_components,
                                        random_state = self.seed )
        self.model.fit(train_dummy)
        
        ### Print that model fit finished and time elapsed
        self.model_fit_finished(start_time, train)

        
    def compute_anomaly_scores(self, data):
        
        def standardize(data, means, sds):
            data_std = data.copy()
            for column in data:
                data_std[column] = (data[column]-means[column]) / sds[column]
                        
            return data_std; 
        
        data_dummy = pd.get_dummies(data)
        data_dummy = standardize(data_dummy, self.means, self.sds)
        """
        p = self.n_components
        weights = (1-self.decay_rate)**(p-1-np.array(range(p)))
        def f(x):
            # can change to <1.0 to save computation and improve score
            return np.sum( (np.array(np.abs(x))*weights)[p-np.int(1.0*p):p] );

        data_pca = pd.DataFrame(self.model.transform(data_dummy))
        scores = data_pca.apply(f, axis=1)
        """
        #data_dummy = data_dummy.drop(self.cols_dropped, axis=1)
        
        try:
            scores = -self.model.score_samples(data_dummy)
        except ValueError:
            scores = data_dummy.shape[0]*[0.5]
            print "WARNING: PPCA model failed, thus 0.5 returned for every sample."
        
        print scores
        return scores;

###############################################################################


###############################################################################
### Calculate input dimension of AE ###########################################
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


############################################################################### 
### Anomaly sampler for URF and UXGB ##########################################
############################################################################### 

def anomaly_sampler(train, frac=0.5, seed=1, verbose=1): 
    
    start_time = time.time() # start clock
    
    ### copy train set with permuted columns
    rd.seed(seed)

    
    train_syn = train.copy()
    train_syn.index = range(train_syn.shape[0])
    for i in range( train_syn.shape[1] ):
        IDX_shuffle = rd.sample( range(train.shape[0]), 
                                 k=int(frac*train.shape[0]) )
        resampled_values = train_syn.ix[IDX_shuffle,i].values
        train_syn.ix[IDX_shuffle,i] = rd.sample( resampled_values, 
                                                 k=len(resampled_values) )
        
    train_syn.index = range(train_syn.shape[0])
    
    time_elapsed = round(time.time() - start_time, 1) # stop clock
            
    ### print AUC score
    if verbose==1:
        print ( '  shuffeling finished (time elapsed: ' 
                + repr(time_elapsed) + 's)\n' )
    
    return train_syn;

###############################################################################           


###############################################################################
### Lexical encoding functions ################################################
###############################################################################

def lexical_enc_feat(feature):
    fature_num = feature.copy()
    fature_num.categories = range(len(feature.categories))
    fature_num = pd.to_numeric(fature_num)
    return fature_num;
    
def lexical_enc_set(data):
    data_num = data.copy()
    for col in data:
        if ( data[col].dtype.name == 'category' ):  # case: categorical
            data_num[col] = lexical_enc_feat(data[col].values)
    return data_num;
    
###############################################################################


###############################################################################
### model parameter plotting functions ########################################
###############################################################################

def print_model_pars(model, model_pars):
    output = ''
    for i in range(len(model_pars)):
        par = model_pars[i]
        output = ( output + 
                   '   ' + (par + 20*' ')[:20] + '= ' 
                   + repr(getattr(model, par)) + '\n'   )
    print output
    
###############################################################################