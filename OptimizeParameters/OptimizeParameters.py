# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 14:44:20 2017

@author: Mathias

This file contains the functions that handles the hyper parameter optimization
of the anomaly detection algorithms.

"""

### load modules
import ast
import configparser
import os
import sys
import time

from bayes_opt import BayesianOptimization


### define directories
OptimizeParameters_dir  = os.path.dirname(__file__)
home_dir                = os.path.normpath(OptimizeParameters_dir 
                                           + os.sep + os.pardir)
AnomalyModels_dir       = os.path.join(home_dir, "AnomalyModels")


### import own modules
sys.path.insert(0, AnomalyModels_dir)
from AnomalyModels import ( IF_AnomalyModel,   URF_AnomalyModel, 
                            UXGB_AnomalyModel, KMD_AnomalyModel, 
                            KMC_AnomalyModel,  AE_AnomalyModel,
                            DAE_AnomalyModel,  OSVM_AnomalyModel,
                            LSAD_AnomalyModel, FRaC_AnomalyModel,
                            GMM_AnomalyModel,  PCA_AnomalyModel )


### config parser 
config = configparser.ConfigParser()
config.read( os.path.join(OptimizeParameters_dir,"config.ini") )


###############################################################################
### function to optimize parameters with bayesian optimization ################
###############################################################################

def OptimizeParameters( 
                        train, test, label_test,
                        AnomalyModel    ='IF',
                        range_pars      = None,
                        init_points     = None,
                        n_iter          = None,
                        acq             = None, 
                        kappa           = None,
                        xi              = None,
                        verbose         = None  ):
    
    init_points = init_points\
        if init_points is not None\
        else int( config['optim pars']['init_points'] )
    
    n_iter = n_iter\
        if n_iter is not None\
        else int( config['optim pars']['n_iter'] )
    
    acq = acq\
        if acq is not None\
        else str( config['optim pars']['acq'] )
        
    kappa = kappa\
        if kappa is not None\
        else float( config['optim pars']['kappa'] )
        
    xi = xi\
        if xi is not None\
        else float( config['optim pars']['xi'] )
        
    verbose = verbose\
        if verbose is not None\
        else int( config['optim pars']['verbose'] )
        
    ### functions
    def print_optim_pars():
        print( 'init_points = ' + repr(init_points) + '\n' +
               'n_iter      = ' + repr(n_iter) + '\n' +
               'acq         = ' + acq + '\n' +
               'kappa       = ' + repr(kappa) + '\n' +
               'xi          = ' + repr(xi) )
        
    def print_range_pars(range_pars):
        print 'model range parameters:'
        for par in range_pars:
            print (3*' ' +par + 20*' ')[:23] + '= ' + repr(range_pars[par])
        print '\n'
        
        
    res = None
    
    if AnomalyModel is None:
        raise Exception('Please specify the parameter "AnomalyModel".')
        
    elif AnomalyModel=='IF':
        if range_pars == None:
            range_pars = ast.literal_eval(config['IF']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print( '=== Optimize Isolation Forest parameters ===' )
            print_optim_pars()
            print_range_pars(range_pars)
        
        def target(n_estimators, sample_frac):
            model = IF_AnomalyModel(  
                                     mode           ='Novelty',
                                     n_estimators   = int(n_estimators), 
                                     sample_frac    = sample_frac,
                                     verbose        = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
        
    
    elif AnomalyModel=='URF':
        if range_pars == None:
            range_pars = ast.literal_eval(config['URF']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize URF parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)

        def target(n_estimators, frac):
            model = URF_AnomalyModel( 
                       mode                 ='Novelty',
                       n_estimators         = int(n_estimators), 
                       #min_impurity_split   = min_impurity_split,
                       #shuffeling           = 'partial',
                       frac                 = frac,
                       verbose              = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;

        
    elif AnomalyModel=='UXGB':
        if range_pars == None:
            range_pars = ast.literal_eval(config['UXGB']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize UXGB parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
           

        def target(nrounds, max_depth, num_parallel_tree, frac):
            model = UXGB_AnomalyModel( 
                       mode                 ='Novelty',
                       nrounds              = int(nrounds), 
                       max_depth            = int(max_depth) ,
                       num_parallel_tree    = int(num_parallel_tree),
                       #shuffeling           = 'partial',
                       frac                 = frac,
                       verbose              = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
    elif AnomalyModel=='KMD':
        if range_pars == None:
            range_pars = ast.literal_eval(config['KMD']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize KMD parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
           

        def target(k, n, max_iterations, subsample, colsample):
            model = KMD_AnomalyModel( 
                       mode                 ='Novelty',
                       k                    = int(k), 
                       n                    = int(n) ,
                       max_iterations       = int(max_iterations),
                       subsample            = float(subsample),
                       colsample            = float(colsample),
                       MSE_normalization    = False,
                       ncores               = None,
                       seed                 = None,
                       verbose              = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
    elif AnomalyModel=='KMC':
        if range_pars == None:
            range_pars = ast.literal_eval(config['KMC']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize KMD parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
           

        def target(k, n, max_iterations, subsample, colsample):
            model = KMC_AnomalyModel( 
                       mode                 ='Novelty',
                       k                    = int(k), 
                       n                    = int(n) ,
                       max_iterations       = int(max_iterations),
                       subsample            = float(subsample),
                       colsample            = float(colsample),
                       MSE_normalization    = False,
                       ncores               = None,
                       seed                 = None,
                       verbose              = 1 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
    elif AnomalyModel=='AE':
        if range_pars == None:
            range_pars = ast.literal_eval(config['AE']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize AE parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
           

        def target(factor, epochs):
            model = AE_AnomalyModel(  
                       mode                 ='Novelty',
                       factor               = factor,
                       epochs               = epochs,
                       ncores               = 6,
                       seed                 = 1,
                       verbose              = 1 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
    elif AnomalyModel=='DAE':
        if range_pars == None:
            range_pars = ast.literal_eval(config['DAE']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize DAE parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
           

        def target(factor, epochs):
            model = DAE_AnomalyModel( 
                       mode                 ='Novelty',
                       factor               = factor,
                       epochs               = epochs,
                       ncores               = 6,
                       seed                 = 1,
                       verbose              = 1 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
            
    elif AnomalyModel=='OSVM':
        if range_pars == None:
            range_pars = ast.literal_eval(config['OSVM']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize OSVM parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
            
        def target(nu, gamma):#, max_samples):
            model = OSVM_AnomalyModel( 
                       mode                 ='Novelty',
                       nu                   = nu, 
                       gamma                = gamma, 
                       max_samples          = 5000,
                       seed                 = 1,
                       verbose              = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
            
    elif AnomalyModel=='LSAD':
        if range_pars == None:
            range_pars = ast.literal_eval(config['LSAD']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize LSAD parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
            
        def target(sigma, rho):
            model = LSAD_AnomalyModel( 
                       mode                 ='Novelty',
                       sigma                = sigma, 
                       rho                  = rho, 
                       seed                 = 1,
                       verbose              = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
        
        
    elif AnomalyModel=='FRaC':
        if range_pars == None:
            range_pars = ast.literal_eval(config['FRaC']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize FRaC parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
            
        def target(n_estimators, bw):
            model = FRaC_AnomalyModel( 
                       mode                 ='Novelty',
                       n_estimators          = int(n_estimators), 
                       bw                    = bw,
                       ncores                = 8, 
                       seed                  = 1,
                       verbose               = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
        
    elif AnomalyModel=='GMM':
        if range_pars == None:
            range_pars = ast.literal_eval(config['GMM']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize GMM parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
            
        def target(n_components):
            model = GMM_AnomalyModel( 
                       mode                  ='Novelty',
                       n_components          = int(n_components), 
                       ncores                = 8, 
                       seed                  = 1,
                       verbose               = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
        
    elif AnomalyModel=='PCA':
        if range_pars == None:
            range_pars = ast.literal_eval(config['PCA']['range_pars'])
        
        ### start clock
        start_time = time.time() 
        
        ### Print begin of performance test
        if verbose == 1:
            print '\n=== Optimize PCA parameters ==='
            print_optim_pars()
            print_range_pars(range_pars)
        
            
        def target(decay_rate):
            model = PCA_AnomalyModel( 
                       mode                  ='Novelty',
                       decay_rate            = decay_rate, 
                       ncores                = 8, 
                       seed                  = 1,
                       verbose               = 0 )
            model.fit(train)
            model.get_performance( test, label_test )
            return model.performance;
        
    else:
        raise Exception(
            'There is no optimization option implemented for the anomaly ' + 
            'model "' + AnomalyModel + '" .' )
    
    ### Optimize model parameters
    bo = BayesianOptimization( target, 
            range_pars )
    bo.maximize(init_points=init_points, n_iter=n_iter, 
                acq=acq, kappa=kappa, xi=xi)
    res = bo.res['max']
        
    ### print best AUC score, corresponding parameters and time elapsed
    print '\n'
    print_optimization_result(res, start_time, verbose)
        
    return res;

###############################################################################        


###############################################################################
### print result ##############################################################
###############################################################################

def print_optimization_result(res, start_time, verbose):
    time_elapsed = round( time.time() - start_time, 1 ) # stop clock
    if verbose == 1:
        pars_opt = res['max_params']
        AUC_opt = res['max_val']
        
        output = ( '> RESULT: (time elapsed: ' 
                   + repr(time_elapsed) + 's)\n' +
                   '   optimal AUC: ' + repr(AUC_opt) 
                   + '\n' +
                   '   optimal parameters:\n' )
            
        for par in pars_opt.keys():
            output = ( output +
                       '      ' + (par + 20 * ' ')[:20] + '= ' 
                       + repr(pars_opt[par]) + '\n' )
        print(output)

###############################################################################
