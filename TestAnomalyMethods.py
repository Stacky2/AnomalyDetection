# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:22:31 2016

@author: Mathias

The file contains the testing method for the anomoaly detection algorithms in 
a novelty detection scenario. To do this it requires the following dependency
files:
    - AnomalyModels: contains the classes for the anomaly models
    - DataSplit: contains the functions that do the split process
    - RepresentationModels: contains the classes for the representation models
    - DataGeneration: contains the functions that generate data if enabled
    - Visualization: contains functions to visualize data
    - OptimizeParameters: contains functions that handle the optimization of
      hyper parameters of anomaly detection methods
The process of the program consists of the following main steps:
    Step 1: Load & split data or generate data
    Step 2: Compute data representation models and transfrom the original data
    Step 3: Test / Optimize the anomaly detection algorithms
    Step 4: Save the results in a log file

"""

###############################################################################
### Loading of packages and config files ######################################
###############################################################################

### import modules
import numpy as np

import os 
import configparser
import ast
import pandas as pd
import sys
import ast

from datetime import datetime

### define directories
home_dir                    = os.path.dirname(__file__)
data_dir                    = os.path.join(home_dir, "data")
AnomalyModels_dir           = os.path.join(home_dir, "AnomalyModels")
AnomalyDataSet_dir          = os.path.join(home_dir, "AnomalyDataSet")
RepresentationModels_dir    = os.path.join(home_dir, "RepresentationModels")
Visualization_dir           = os.path.join(home_dir, "Visualization")
DataGeneration_dir          = os.path.join(home_dir, "DataGeneration")
OptimizeParameters_dir      = os.path.join(home_dir, "OptimizeParameters")
Log_dir                     = os.path.join(home_dir, "Log")

### import own modules
## import the AnomalyModel classes
sys.path.insert(0, AnomalyModels_dir)
from AnomalyModels import ( IF_AnomalyModel,   URF_AnomalyModel, 
                            UXGB_AnomalyModel, KMD_AnomalyModel, 
                            KMC_AnomalyModel,  AE_AnomalyModel,
                            DAE_AnomalyModel,  OSVM_AnomalyModel,
                            LSAD_AnomalyModel, FRaC_AnomalyModel,
                            GMM_AnomalyModel,  PCA_AnomalyModel )

sys.path.insert(0, AnomalyDataSet_dir)
from AnomalyDataSet import AnomalyDataSet


sys.path.insert(0, RepresentationModels_dir)
from RepresentationModels import ( ID_RepModel, 
                                   PCA_RepModel,
                                   FICA_RepModel,
                                   TSVD_RepModel,
                                   AE_RepModel,
                                   EE_RepModel ) 

#sys.path.insert(0, DataGeneration_dir)
#from DataGeneration import generateNoveltyData  

sys.path.insert(0, Visualization_dir)
from Visualization import plot_samples_TSNE

sys.path.insert(0, OptimizeParameters_dir)
from OptimizeParameters import OptimizeParameters

###############################################################################



###############################################################################
### Functions #################################################################
###############################################################################

def start(dataset, mode):
    print( '\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% ================================================= %%%\n' +
           '%%% === START - TEST OF ' + mode.upper() 
           + ' DETECTION METHODS === %%%\n' +
           '%%% ================================================= %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' 
            )
    print( 'Dataset: ' + dataset + '\n' )
    
    
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


def print_load_data():
    
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 1a: Load Data %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )

    
def print_SplitData():
    
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 1b: Split Data %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )

    
"""
def print_simulate_data():
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 1: Simulate Datasets %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )
""" 
        
def FeatureRepresentation(train, test, config, n_models=1, 
                          visualize_test=False):
    
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 2: Compute Feature Representation %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )
    
    global RepModels
    
    RepModels = []
    
    ### Initialize train and test representation datasets
    train_rep = pd.DataFrame( np.nan, 
                              index   = range(train.shape[0]), columns = [] )
    test_rep  = pd.DataFrame( np.nan, 
                              index   = range(test.shape[0]), columns = [] )
    
    ### Compute all representation models in the list models
    for i in range(1,n_models+1):
        RepModel = None
        
        if 'Rep'+repr(i) in config.keys():
            model_type = ast.literal_eval(config['Rep'+repr(i)]['model_type'])
        else:
            raise Exception( 'There is no configuration key for Rep' 
                             + repr(i) + ' given in the main config file!' )
            
        if 'params' in config['Rep'+repr(i)]:
            params = ast.literal_eval(config['Rep'+repr(i)]['params'])
        else:
            params = {}
            
        ### check if the method define by model_type exists, 
        ### eg. if model_type = 'PCA', check if 'PCA_RepModel' exists        
        try:
            RepModel = eval(model_type+'_RepModel')()
            print( 3*'%'+' ' + RepModel.model_type + ' ' + (3*'%') + '\n'
                   + (len(RepModel.model_type)+2*4)*'%' )
            RepModel.set_params(**params)
            RepModel.fit(train)
            train_rep = pd.concat( [ train_rep, 
                                     RepModel.get_representation(train) ], 
                                     axis=1 )
            test_rep  = pd.concat( 
                           [ test_rep, 
                             RepModel.get_representation(test, test=True) ], 
                           axis=1 )
            
        except NameError:
            raise Exception('There exists no method with the name ' 
                             + model_type + '_RepModel in this directory')
            
        
        RepModels.append(RepModel)
        
        print( '\n' )
        
    if visualize_test:
        print 'Visulaize test set using TSNE...\n'
        plot_samples_TSNE(test, label_test, nsamples=2000)        
        
    return train_rep, test_rep;
    
    
def TestAnomalyModel(train, test, label_test, config, mode, n_models=1):
    
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 3: Test '+ mode +' Detection Methods %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )

    
    models = []
    
    
    for i in range(1,n_models+1):
        
   
        if 'Anom'+repr(i) in config.keys():
            model_type = ast.literal_eval(config['Anom'+repr(i)]['model_type'])
        else:
            raise Exception( 'There is no configuration key for Anom' 
                             + repr(i) + ' given in the main config file!' )
                
        if 'params' in config['Anom'+repr(i)]:
            params = ast.literal_eval(config['Anom'+repr(i)]['params'])
        else:
            params = {}
        
        ### check if the method define by model_type exists, 
        ### eg. if model_type = 'IF', check if 'IF_AnomalyModel' exists    
        try:
            model = eval(model_type+'_AnomalyModel')(mode=mode)
            print( 3*'%'+' ' + model.model_type + ' model ' + (3*'%') + '\n'
                   + (len(model.model_type)+2*4+6)*'%' )
            model.set_params(**params)
            model.fit(train)
            model.get_performance(test, label_test) #, print_rd_scores = 100) 
            
        except NameError:
            raise Exception('There exists no method with the name ' 
                             + model_type + '_AnomalyModel in this directory')
        
        models.append(model)
        
    return models;
    
    
def OptimizeAnomalyModels(train, test, label_test, config, mode, n_models=1):
    
    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 3: Optimize '+mode+' Detection Methods %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )
    
    results = []
    
    for i in range(1,n_models+1):
        
        if 'Anom'+repr(i) in config.keys():
            model_type = ast.literal_eval(config['Anom'+repr(i)]['model_type'])
        else:
            raise Exception( 'There is no configuration key for Anom' 
                             + repr(i) + ' given in the main config file!' )
            
        if 'params' in config['Anom'+repr(i)]:
            print( 'Warning: variable params is not used ' +
                   'because of optimize=True.' )
            
        ### check if the method define by model_type exists, 
        ### eg. if model_type = 'IF', check if 'IF_AnomalyModel' exists  
        try:
            model = eval(model_type+'_AnomalyModel')(mode=mode)
            print( 3*'%'+' ' + model.model_type + ' model ' + (3*'%') + '\n'
                  + (len(model.model_type)+2*4+6)*'%' )
            res = OptimizeParameters( train, test, label_test,
                                     AnomalyModel=model_type, verbose = 1 )
            
        except NameError:
            raise Exception('There exists no method with the name ' 
                             + model_type + '_AnomalyModel in this directory')
        
        results.append(res)
        
    return results;

    
def SaveResults(AnomModels, AnomalyDataSet, config, config_data, train, test):

    print( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' +
           '%%% STEP 4: Save Results %%%\n' +
           '%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' )
    
    ### functions for different output parts ###
    ############################################
    def output_dataset(dataset, config_data):
        output = ( '<<< Dataset >>>\n' +
                   '===============\n' +
                   'name           = ' + dataset + '\n' +
                   'nrow test set  = ' + repr(test.shape[0]) + '\n' +
                   'nrow train set = ' + repr(train.shape[0]) + '\n' +
                   'anomaly ratio  = ' 
                   + repr(float(config_data[dataset]['anom_ratio'])) + '\n' +  
                   'split seed     = ' + repr(AnomalyDataSet.seed) + '\n\n' )
        return output;
    
        
    def model_results(AnomalyModel):
        output = ( '>>> ' + AnomalyModel.model_type + ' <<<\n' +
                   (len(AnomalyModel.model_type)+8) * '=' + '\n' +
                   'AUC:        ' + repr(AnomalyModel.performance) + '\n' +
                   'fit time:   ' + repr(AnomalyModel.fit_time) + '\n' +
                   'score time: ' + repr(AnomalyModel.score_time) + '\n' )
        
        output = ( output +
                   'model parameters:\n' )
        for i in range(len(AnomalyModel.model_pars)):
            par = AnomalyModel.model_pars[i]
            output = ( output 
                       + '   ' + (par + 20 *' ')[:20] + ' = ' 
                       + repr(getattr(AnomalyModel, par)) + '\n')
        output = ( output + '\n' )
        
        return output;
        
        
    def optim_results(n_models, results):
        output = ''
        for i in range(n_models):
            res = results[i]
            pars_opt = res['max_params']
            AUC_opt = res['max_val']
            model_type = ast.literal_eval(
                            config['Anom'+repr(i+1)]['model_type'])
            
            output = ( output + 
                       '>>> ' + model_type + ' <<<\n' +
                       '   optimal AUC: ' + repr(AUC_opt) + '\n' +
                       '   optimal parameters:\n' )
            
            output_pars = ''
            for par in pars_opt.keys():
                output_pars = ( output_pars +
                           '      ' + (par + 20 * ' ')[:20] + '= ' 
                           + repr(pars_opt[par]) + '\n' )
                
            output = output + output_pars + '\n'
            
        return output
    
        
    def output_RepModels(RepModels):
        n_features = 0
        fit_time   = 0
        trafo_time = 0
        for i in range(len(RepModels)):
            n_features += RepModels[i].n_features
            fit_time += RepModels[i].fit_time
            trafo_time += RepModels[i].trafo_time
            
        output = ( '||| Representation |||\n' +
                   '======================\n' +
                   'total number of features = ' + repr(n_features) + '\n' +
                   'fit time                 = ' + repr(fit_time) + '\n' +
                   'test trafo time          = ' + repr(trafo_time) + '\n' )
        
        for i in range(len(RepModels)):
            output_model = ( 
                       '- ' + RepModels[i].model_type 
                       + ' ( ' + repr(RepModels[i].n_features)
                       + ' features )\n' +
                       '  fit time:        ' + repr(RepModels[i].fit_time) 
                       + '\n' +
                       '  test trafo time: ' + repr(RepModels[i].trafo_time) 
                       + '\n' )
            
            for j in range(len(RepModels[i].model_pars)):
                output_model = ( output_model +
                           '     ' 
                           +  (RepModels[i].model_pars[j] + 20*' ')[:20] +
                           '= ' + repr(getattr( RepModels[i], 
                                                RepModels[i].model_pars[j] )) 
                           + '\n' )
            
            output = output + output_model + '\n'
        
        output = output + '\n'
            
        return output;
    ############################################
    
    ### generate different outputs ###
    ##################################
    
    ### output dataset info
    dataset = ast.literal_eval(config['Data']['dataset'])
    output_dataset = output_dataset(dataset, config_data)
    
    ### output representation models info
    output_RepModels = output_RepModels(RepModels)
    
    ### output anomaly models info
    if ast.literal_eval(config['AnomalyMethods']['optimize']):
        output_models = optim_results(n_AnomModels, results)
    else:
        output_models = ''
        for model in AnomModels:
            output_models = output_models + model_results(model)
        
    ##################################
        
        
    ### combine outputs and write output file ###
    #############################################
    mode = ast.literal_eval(config['Data']['mode'])
    output = ( '=============================================\n' +
               '=== Performance Sheet - '+mode+' Detection ===\n' +
               '=============================================\n\n' +
               output_dataset +
               output_RepModels +
               output_models )
    
    name_output_file = ( 'Results TestNoveltyMethods ' 
                         + datetime.now().strftime('%Y-%m-%d %H-%M-%S') 
                         + '.txt' )
    outputfile_dir = os.path.join(Log_dir, name_output_file)
    text_file = open(outputfile_dir, "w")
    text_file.write(output)
    text_file.close()
    
    print 'Results saved in file: "' + name_output_file + '".'
    #############################################
    
###############################################################################
    


###############################################################################
### MAIN PROGRAM ##############################################################
###############################################################################

def execute_job(config_name):

    ### define config parsers
    config = configparser.ConfigParser()
    config.read( os.path.join(home_dir, 'configs',config_name  ))
    
    dataset = ast.literal_eval(config['Data']['dataset'])
    config_data = configparser.ConfigParser()
    config_data.read(os.path.join(data_dir, "config.ini")) 
    
    mode = ast.literal_eval(config['Data']['mode'])
    
    start(dataset, mode=mode)
    
    ### Get data accoring to the specified source
    if str(config['Data']['source']) == "simulated":
        raise Exception('Data Simulation not implemented yet!')
        #train_raw, test_raw, label_test = simulate_data(dataset=dataset)
        
    elif str(config['Data']['source']) == "original":
        ### load and split the data
        print_load_data()
        myAnomalyDataSet = AnomalyDataSet(dataset = dataset, mode = mode)
        myAnomalyDataSet.set_params_by_config(config_data)
        params_DataSet = ast.literal_eval(config['Data']['params'])
        myAnomalyDataSet.set_params(**params_DataSet)
        myAnomalyDataSet.load_data_from_folder(home_dir)
        
        print_SplitData()
        train_raw, test_raw, label_test = myAnomalyDataSet.getAnomalySplit()
    
    ### Compute Feature Representation
    n_RepModels=ast.literal_eval(config['Representation']['n_models'])
    train, test = FeatureRepresentation(
                     train_raw, test_raw, config, 
                     n_models=n_RepModels,
                     visualize_test=ast.literal_eval( 
                                           config['Data']['visualize_test']) )
    
    ### Test AnomalyModels
    n_AnomModels = ast.literal_eval(config['AnomalyMethods']['n_models'])
    if ast.literal_eval(config['AnomalyMethods']['optimize']):
        results = OptimizeAnomalyModels(
                    train, test, label_test, 
                    n_models=n_AnomModels, config=config, mode = mode)
    else:
        AnomModels = TestAnomalyModel(
                        train, test, label_test, 
                        n_models=n_AnomModels, config=config, mode = mode) 
    
    ### Save results
    SaveResults(AnomModels,myAnomalyDataSet, config, config_data, train, test)

    return AnomModels;

###############################################################################


if __name__ == "__main__":
    results = execute_job("config_auto.ini")