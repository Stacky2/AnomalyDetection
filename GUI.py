# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:03:48 2017

@author: Mathias

This file contains the program for Graphical User Inerface which allowes an
easy useage of the test function TestAnomalyMethods. Using the different 
button and entry fields one can specify the details of a test run. After 
clicking the run button, the program automatically generates a config file and
runs the testing function. After that, the results are displayed in a 
comparison window.

"""

### import modules
from tkinter import ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

import Tkinter as tk
import numpy as np
import pandas as pd
import tkMessageBox
import configparser
import os
import math
import sys
import ttk
import ast
import time


### define directories
home_dir                    = os.path.dirname(__file__)
data_dir                    = os.path.join(home_dir, "data")
configs_dir                 = os.path.join(home_dir, "configs")
AnomalyModels_dir           = os.path.join(home_dir, "AnomalyModels")
Visualization_dir           = os.path.join(home_dir, "Visualization")
RepModels_dir               = os.path.join(home_dir, "RepresentationModels")
AnomalyDataSet_dir          = os.path.join(home_dir, "AnomalyDataSet")



### laod own modules
from TestAnomalyMethods import execute_job

sys.path.insert(0, AnomalyDataSet_dir)
from AnomalyDataSet import AnomalyDataSet

from AnomalyModels import ( IF_AnomalyModel,   URF_AnomalyModel, 
                            UXGB_AnomalyModel, KMD_AnomalyModel, 
                            KMC_AnomalyModel,  AE_AnomalyModel,
                            DAE_AnomalyModel,  OSVM_AnomalyModel,
                            LSAD_AnomalyModel, FRaC_AnomalyModel,
                            GMM_AnomalyModel,  PCA_AnomalyModel )


sys.path.insert(0, RepModels_dir)
from RepresentationModels import ( ID_RepModel, 
                                   PCA_RepModel,
                                   FICA_RepModel,
                                   TSVD_RepModel,
                                   AE_RepModel,
                                   EE_RepModel ) 

#sys.path.insert(0, Visualization_dir)
from Visualization import grad_hex_colors, grad_hex_colors_inv


### define config parsers
config_data = configparser.ConfigParser()
config_data.read(os.path.join(data_dir, "config.ini"))    

config_AD = configparser.ConfigParser()
config_AD.read(os.path.join(AnomalyModels_dir, "config.ini"))    
    
config_Rep = configparser.ConfigParser()
config_Rep.read(os.path.join(RepModels_dir,'config.ini'))


### define sets for possible datasets, AD methods and RepModels
datasets = [str(dataset) 
            for dataset in config_data.keys()[1:len(config_data.keys())]]
ADmethods = ["IF","URF","UXGB","KMD","KMC","AE","DAE","OSVM","LSAD","FRaC",
             "GMM","PCA"]
RepModels = ["ID","PCA","FICA","TSVD","AE","EE"]


### extract the different params for the AD & Rep methods
ADparams = {}
for ID in ADmethods:
    model = eval(ID+'_AnomalyModel')(mode="Novelty")
    ADparams[ID] = model.model_pars
            
RepModels_names = {}
for ID in RepModels:
    model = eval(ID+'_RepModel')()
    RepModels_names["["+ID+"] "+model.model_type] = ID


###############################################################################
###############################################################################
###############################################################################

class Settings_App2():
    
    ADmethods_sel       = {}
    n_Reps              = None 
    dataset_sel         = None
    LF_Rep              = None
    widgets_RepModels   = []
    RepModels           = RepModels
    datasets            = datasets
    ndatasets           = len(datasets)
    ADmethods           = ADmethods
    nADs                = len(ADmethods)
    RepModels_sel       = []
    RepModels_params    = []
    config_data         = config_data
    ADparams            = ADparams
    ADparams_values     = {}
    RepModels_names     = RepModels_names
    
    for method in ADmethods:
        ADmethods_sel[method] = False
    
    
    def __init__(self, root):
        self.master = ttk.Notebook(root)
        self.createWidgets()
        self.master.pack(expand=1, fill="both")
        
    def createWidgets(self):
        self.createPage1()
        self.createPage2()
        self.createPage3()
        self.createPage4()
        self.createPage5()
        self.createPage6()
        
    def createPage1(self):
        self.page1 = ttk.Frame(self.master)
        self.createContentPage1()
        self.master.add(self.page1, text=' Mode ')
        
    def createPage2(self):
        self.page2 = ttk.Frame(self.master)
        self.createContentPage2()
        self.master.add(self.page2, text=' Data ')
        
    def createPage3(self):
        self.page3 = ttk.Frame(self.master)
        self.createContentPage3()
        self.master.add(self.page3, text=' Representation ')
        
    def createPage4(self):
        self.page4 = ttk.Frame(self.master)
        self.createContentPage4()
        self.master.add(self.page4, text=' Anomaly Detection ')
        
    def createPage5(self):
        self.page5 = ttk.Frame(self.master)
        self.createContentPage5()
        self.master.add(self.page5, text=' Run ')
        
    def createPage6(self):
        self.page6 = ttk.Frame(self.master)
        self.createContentPage6()
        self.master.add(self.page6, text=' Info ')
        
###############################################################################

    def createContentPage1(self):
        self.LF_mode = tk.LabelFrame(self.page1, text="Mode")
        self.LF_mode.pack(fill=tk.BOTH, expand=1)
        self.createModeSelection()
        
        
    def createModeSelection(self):
        self.L_select_mode = tk.Label(self.LF_mode, 
                                      text = "Select the outlier mode:")
        self.L_select_mode.grid(row=0, sticky=tk.W)
        

        
        self.var_mode= tk.IntVar()
        self.R_mode1 = tk.Radiobutton(
                                self.LF_mode, 
                                text = 'Novelty', 
                                variable = self.var_mode, 
                                value = 0,
                                command = self.command_mode)
        self.R_mode1.grid(row=0, column=1, sticky=tk.W)
        self.R_mode2 = tk.Radiobutton(
                                self.LF_mode, 
                                text = 'Outlier', 
                                variable = self.var_mode, 
                                value = 1,
                                command = self.command_mode)
        self.R_mode2.grid(row=0, column=2, sticky=tk.W)
        
        
    def command_mode(self):
        self.RefreshADdefaults()
        
###############################################################################

    def createContentPage2(self):
        self.createDatasetSelection()
        self.createDataParamsSelection()

        
    def createDatasetSelection(self):
        self.F_datasets = tk.LabelFrame(self.page2)
        self.F_datasets.pack(fill=tk.BOTH, expand=1)
        
        self.F_datasets_opts = tk.LabelFrame(self.F_datasets)
        self.F_datasets_opts.grid(row=0, column=0, 
                                  sticky = tk.N+tk.S+tk.W+tk.E)
        
        self.L_select_data = tk.Label(self.F_datasets_opts, 
                                      text = "Select a dataset:")
        self.L_select_data.grid(row=0, sticky=tk.W)
        
        self.var_datasets = tk.IntVar()

        self.F_datasets_real = tk.LabelFrame(self.F_datasets_opts, 
                                             text="real data")
        self.F_datasets_real.grid(row=1, sticky = tk.W+tk.E)

        def command_dataset_sel():
            self.dataset_sel = self.datasets[self.var_datasets.get()-1]
            self.var_ntrain_max.set(
                    self.config_data[self.dataset_sel]['n_train_samples_max'])
            self.var_ratio.set( 
                    self.config_data[self.dataset_sel]['anom_ratio'])
            self.var_ntest.set( 
                    self.config_data[self.dataset_sel]['n_test_samples'])
            self.var_seed_data.set( self.config_data[self.dataset_sel]['seed'])
            self.var_values_anom.set( 
                    ast.literal_eval(
                            self.config_data[self.dataset_sel]['values_anom']) 
                    )
            self.var_values_norm.set( 
                    ast.literal_eval(
                            self.config_data[self.dataset_sel]['values_norm']) 
                    )
            if hasattr(self, 'TSNE_label'):
                self.TSNE_label.grid_forget()
            
            try: 
                image_path = os.path.join(
                        os.path.join(data_dir,self.dataset_sel ), "TSNE.png")
                self.image = Image.open(image_path)
                size = (512,512)
                self.image.thumbnail(size, Image.ANTIALIAS)
                self.photo = ImageTk.PhotoImage(self.image)
                self.TSNE_label = tk.Label(self.F_datasets_pict, 
                                           image=self.photo)
            except:
                self.TSNE_label = tk.Label(self.F_datasets_pict, 
                                           text='[No picture avilable]')
                
            self.TSNE_label.grid()
        
        
        self.ndatasets=len(self.datasets)
        
        self.R_datasets = self.ndatasets * [None] 
        for i in range(self.ndatasets):
            self.R_datasets[i] = tk.Radiobutton(
                                    self.F_datasets_real, 
                                    text = self.datasets[i], 
                                    variable = self.var_datasets, 
                                    value = (i+1),
                                    command = command_dataset_sel)
            self.ncols_datasets = int(3)
            self.nrows_datasets = int(math.ceil(float(self.ndatasets)/
                                                self.ncols_datasets))
            #R.pack( anchor = W )
            self.R_datasets[i].grid(
                    row     = (int(i)%self.nrows_datasets)+1, 
                    column  = int(i)/self.nrows_datasets , 
                    sticky  = tk.W )
            
        F_datasets_real = tk.LabelFrame(self.F_datasets_opts, 
                                        text="simulated data")
        F_datasets_real.grid(row=2, sticky = tk.W+tk.E)
        L = tk.Label(F_datasets_real, 
                     text = "[Add simulation datasets]")
        L.grid()
        
        self.F_datasets_pict = tk.LabelFrame(self.F_datasets)
        self.F_datasets_pict.grid(row=0, column=1)
        
        
    def createDataParamsSelection(self): 
        self.F_DataParams = tk.LabelFrame(self.page2)
        self.F_DataParams.pack(fill=tk.BOTH, expand=1)
        
        L = tk.Label(self.F_DataParams, text = "Select dataset parameters:")
        L.grid(row=0, sticky=tk.W)
        
        self.var_ntest = tk.StringVar(self.master)
        self.var_ntest.set("10000")        
        self.L_ntest = tk.Label(self.F_DataParams, text = "test size:")
        self.L_ntest.grid(row=1, column=0, sticky=tk.W)
        self.S_ntest = tk.Spinbox(self.F_DataParams, from_=1, to=10000000, 
                                  textvariable=self.var_ntest)
        self.S_ntest.grid(row=1, column=1, sticky=tk.E)
        
        self.var_ntrain_max = tk.StringVar(self.master)
        self.var_ntrain_max.set("10000")
        self.L_select_data = tk.Label(self.F_DataParams, 
                                      text = "maximal train size:")
        self.L_select_data.grid(row=2, column=0, sticky=tk.W)
        self.S_ntrain_max = tk.Spinbox(self.F_DataParams, from_=1, to=10000000, 
                                       textvariable=self.var_ntrain_max)
        self.S_ntrain_max.grid(row=2, column=1, sticky=tk.E)
        
        self.var_values_norm = tk.StringVar(self.master)
        L_values_norm= tk.Label(self.F_DataParams, text = "normal values:")
        L_values_norm.grid(row=3, column=0, sticky=tk.W)
        E_values_norm= tk.Entry(self.F_DataParams, bd =5, 
                                textvariable=self.var_values_norm)
        E_values_norm.grid(row=3, column=1, sticky=tk.E)
        
        self.var_values_anom = tk.StringVar(self.master)
        L_values_anom= tk.Label(self.F_DataParams, text = "anomaly values:")
        L_values_anom.grid(row=4, column=0, sticky=tk.W)
        E_values_anom= tk.Entry(self.F_DataParams, bd =5, 
                                textvariable=self.var_values_anom)
        E_values_anom.grid(row=4, column=1, sticky=tk.E)
        
        self.var_ratio = tk.StringVar(self.master)
        self.var_ratio.set("0.1")
        self.L_sel_ratio = tk.Label(self.F_DataParams, text = "anomaly ratio:")
        self.L_sel_ratio.grid(row=5, column=0, sticky=tk.W)
        self.E_ratio = tk.Entry(self.F_DataParams, bd =5, 
                                textvariable=self.var_ratio)
        self.E_ratio.grid(row=5, column=1, sticky=tk.E)
        
        self.var_seed_data = tk.StringVar(self.master)
        self.var_seed_data.set("1")
        L_seed = tk.Label(self.F_DataParams, text = "data seed:")
        L_seed.grid(row=6, column=0, sticky=tk.W)
        E_seed = tk.Entry(self.F_DataParams, bd =5, 
                          textvariable=self.var_seed_data)
        E_seed.grid(row=6, column=1, sticky=tk.E)
        
        self.var_nnoise = tk.StringVar(self.master)
        self.var_nnoise.set('0')
        L_nnoise = tk.Label(self.F_DataParams, 
                            text = "number of noise features:")
        L_nnoise.grid(row=7, column=0, sticky=tk.W)
        E_nnoise = tk.Entry(self.F_DataParams, bd =5, 
                          textvariable=self.var_nnoise)
        E_nnoise.grid(row=7, column=1, sticky=tk.E)
        
        B_overview = tk.Button(self.F_DataParams, text = "Print overview")
        B_overview['command'] = self.show_overview
        B_overview.grid(row=8, column=1, sticky=tk.E+tk.W)
        
    def show_overview(self):
        self.overviewWindow = tk.Toplevel(self.master)
        self.overview_App = Overview_App(self.overviewWindow, self.dataset_sel)
        
###############################################################################
        
    def createContentPage3(self):
        self.createRepSelection()
        
    def createRepSelection(self):
        ## create labelframe for visivility
        self.F_Rep = tk.LabelFrame(self.page3, text="Representation Models")
        self.F_Rep.pack(fill=tk.BOTH, expand=1)
        
        ## describe what to choose
        self.L_choose_Rep = tk.Label(
                self.F_Rep, 
                text="Choose the number of representation models.")
        self.L_choose_Rep.grid(row=0, column=0,  sticky=tk.W)
        
        ## to specify the number of representation models
        self.var_nReps = tk.StringVar(self.master)
        self.S_n_Reps = tk.Spinbox(self.F_Rep, from_=1, to=4, 
                                   textvariable=self.var_nReps) 
        
        self.S_n_Reps.grid(row=0, column=1, sticky=tk.E)
        self.n_Reps = self.S_n_Reps
        
        ## refresh button
        self.B_refresh_Reps = tk.Button(self.F_Rep, text = "Refresh models")
        self.B_refresh_Reps['command'] = self.refresh_RepModels
        self.B_refresh_Reps.grid(row=0, column=3, sticky=tk.E)

        self.refresh_RepModels() # to have an initial Rep model displayed

###############################################################################
              
    def refresh_RepModels(self):
        for widget in self.widgets_RepModels: # clear for new Rep models number
            widget.pack_forget()
            widget.grid_forget()
            self.RepModels_sel = []
            self.RepModels_params = []
        
        self.LFs_Reps        = int(self.var_nReps.get()) * [None]
        self.var_Reps        = int(self.var_nReps.get()) * [None]
        self.W_Reps          = int(self.var_nReps.get()) * [None]
        self.L_Reps          = int(self.var_nReps.get()) * [None]
        self.L_Reps_factor   = int(self.var_nReps.get()) * [None]
        self.E_Reps          = int(self.var_nReps.get()) * [None]
        self.E_Reps_factor   = int(self.var_nReps.get()) * [None]
        self.var_Reps_factor = ( int(self.var_nReps.get()) 
                                 * [tk.StringVar(self.master)] )
        
        for i in range(int(self.var_nReps.get())):
            # label frame used for single RepModel
            self.LFs_Reps[i] = tk.LabelFrame(self.F_Rep, 
                                  text = "RepModel "+ repr(i+1))
            self.LFs_Reps[i].grid()

            self.var_Reps[i] = tk.StringVar(self.LFs_Reps[i])
            self.var_Reps[i].set('ID')# default value
            
            def setRepDefault(*args): 
                self.var_Reps_factor[i].set(
                        config_Rep[self.var_Reps[i].get()]['factor'])
                         
                
            self.W_Reps[i] = tk.OptionMenu(self.LFs_Reps[i], self.var_Reps[i], 
                       *self.RepModels, command=setRepDefault)
            self.W_Reps[i].grid(row=0, column=0, sticky=tk.N)
            self.RepModels_sel.append(self.var_Reps[i])
            
            self.var_Reps_factor[i].set(
                    config_Rep[self.var_Reps[i].get()]['factor']) 
            
            self.L_Reps_factor[i] = tk.Label(self.LFs_Reps[i], text="factor")
            self.L_Reps_factor[i].grid(row=1, column=0, sticky=tk.W)
            self.E_Reps_factor[i] = tk.Entry(self.LFs_Reps[i], bd =5 , 
                                       textvariable=self.var_Reps_factor[i])
            self.E_Reps_factor[i].grid(row=1, column=1, sticky=tk.W) 
            self.L_Reps[i] = tk.Label(self.LFs_Reps[i], text="prameters")
            self.L_Reps[i].grid(row=2, column=0, sticky=tk.W)
            self.E_Reps[i] = tk.Entry(self.LFs_Reps[i], bd =5)
            self.E_Reps[i].grid(row=2, column=1, sticky=tk.W)

            
            ### append widgets to a list st they can be removed when refreshing
            self.widgets_RepModels.append(self.LFs_Reps[i])
            self.widgets_RepModels.append(self.W_Reps[i])
            self.widgets_RepModels.append(self.L_Reps[i])
            self.widgets_RepModels.append(self.L_Reps_factor[i])
            self.widgets_RepModels.append(self.E_Reps[i])
            self.widgets_RepModels.append(self.E_Reps_factor[i])
            
        
###############################################################################
        
    def createContentPage4(self):
        self.createAnomalyAlgoSelection()
        
    def createAnomalyAlgoSelection(self):
        self.F_ADalgos = tk.LabelFrame(self.page4, 
                                        text="Anomaly Detection Algorithms")
        self.F_ADalgos.pack(fill=tk.BOTH, expand=1)
        
        
        self.L_choose_ADalgo = tk.Label(
                self.F_ADalgos, 
                text="Choose anomaly detection algorithms to test.")
        self.L_choose_ADalgo.grid(row = 0, column=0) #.pack()

        self.LFs_ADalgos = int(len(self.ADmethods)) * [None]


        self.ncols_ADs = int(4)
        self.nrows_ADs = int(math.ceil(float(self.nADs)/
                                                self.ncols_ADs))


        for i in range(len(self.ADmethods)):
            
            method = self.ADmethods[i]
            
            temp = eval(method+'_AnomalyModel')(mode="Novelty")
            model_type = temp.model_type
            # label frame used for single RepModel
            self.LFs_ADalgos[i] = tk.LabelFrame(self.F_ADalgos, 
                                    text = model_type) 
            self.LFs_ADalgos[i].grid(                    
                    row     = (int(i)%self.nrows_ADs)+1, 
                    column  = int(i)/self.nrows_ADs , 
                    sticky  = tk.N+tk.S+tk.E+tk.W)
            
            self.var_AD = tk.IntVar()
            C = tk.Checkbutton( self.LFs_ADalgos[i], 
                                text = method, variable = self.var_AD,
                                onvalue = True, offvalue = False ) 
                                #, command = sel(method))
            self.ADmethods_sel[method] = self.var_AD
            row = 0
            C.grid(row=row, column=0) #.pack()
            row += 1
            
            params_values = {}
            
            for param in self.ADparams[method]:
                L = tk.Label(self.LFs_ADalgos[i], text=param)
                L.grid(row=row, column=0)
                
                var_param = tk.StringVar(self.master)
                params_values[param] = var_param
                E = tk.Entry(self.LFs_ADalgos[i], bd =5, 
                             textvariable = params_values[param] ) 
                             #, textvariable= ???)
                E.grid(row=row, column=1)
                
                row +=1
            
            self.ADparams_values[method] = params_values
                                
        self.RefreshADdefaults()
        
                                
    def RefreshADdefaults(self):
        for i in range(len(self.ADmethods)):
            
            method = self.ADmethods[i]
            mode_short = "_Nov" if self.var_mode.get() == 0 else "_Out"
            
            for param in self.ADparams[method]:
                self.ADparams_values[method][param].set(
                        ast.literal_eval(config_AD[method+mode_short][param]))
            
###############################################################################
    
    def createContentPage5(self):
        self.createConfigSettingsSelection()
        self.createResponseButton()
        
    def createConfigSettingsSelection(self):
        self.F_ConfigSettings = ttk.Frame(self.page5)
        self.F_ConfigSettings.grid()
        
        ### selection of config file name
        self.var_configName = tk.StringVar(self.master)
        self.var_configName.set('config_auto')
        L = tk.Label(self.F_ConfigSettings, text="config file name:")
        L.grid(row = 0, column=0)
        E = tk.Entry(self.F_ConfigSettings, bd =5, 
                     textvariable = self.var_configName )
        E.grid(row=0, column=1)
        
        ### selection of batch name
        self.var_batchName = tk.StringVar(self.master)
        self.var_batchName.set('myBatch')
        L_batch = tk.Label(self.F_ConfigSettings, text="batch name:")
        L_batch.grid(row = 1, column=0)
        E_batch = tk.Entry(self.F_ConfigSettings, bd =5, 
                     textvariable = self.var_batchName ) 
        E_batch.grid(row=1, column=1)
        
        ### selection of config file description
        self.var_desc = tk.StringVar(self.master)
        self.var_desc.set('')
        L_desc = tk.Label(self.F_ConfigSettings, 
                          text="description in config file:")
        L_desc.grid(row = 2, column=0)
        E_desc = tk.Entry(self.F_ConfigSettings, bd =5, 
                     textvariable = self.var_desc ) 
        E_desc.grid(row=2, column=1)
        
        
    def createResponseButton(self):
        self.createSaveButton()
        self.createSaveInBatchButton()
        self.createRunBatchButton()
        self.createRunButton()
        
    def createSaveButton(self):
        B_save = tk.Button(self.page5, text="Save config")
        B_save['command'] = self.response_save
        B_save.grid(sticky=tk.N+tk.S+tk.W+tk.E)
        
    def response_save(self):
        self.WriteSave_config()
        
    def createSaveInBatchButton(self):
        B_save = tk.Button(self.page5, text="Save config in batch")
        B_save['command'] = self.response_save_batch
        B_save.grid(sticky=tk.N+tk.S+tk.W+tk.E)

    def response_save_batch(self):
        
        if type(eval(self.var_seed_data.get()))==list:
            print 'CASE: list of seeds'
            
            for seed in eval(self.var_seed_data.get()):

                self.write_config(seed)
                
                #self.save_config_batch(file_name)
                name_output_file = self.var_configName.get()
                batch_name = self.var_batchName.get()
                outputfile_dir = os.path.join(configs_dir, batch_name)
                file_path = os.path.join(configs_dir,self.var_batchName.get(), 
                               name_output_file+"_s"+repr(seed)+".ini")
                if not os.path.exists(outputfile_dir):
                    os.makedirs(outputfile_dir)
                text_file = open(file_path, "w")
                text_file.write(self.config)
                text_file.close()
                
        else:
            self.write_config()
            #self.save_config_batch()
            
            name_output_file = self.var_configName.get()
            batch_name = self.var_batchName.get()
            outputfile_dir = os.path.join(configs_dir, batch_name)
            file_path = os.path.join(configs_dir,self.var_batchName.get(), 
                                     name_output_file+".ini")
            if not os.path.exists(outputfile_dir):
                os.makedirs(outputfile_dir)
            text_file = open(file_path, "w")
            text_file.write(self.config)
            text_file.close()
            
        
        batch_name = self.var_batchName.get()
        outputfile_dir = os.path.join(configs_dir, batch_name)
        print ( 'Files in batch "' + batch_name + ': ' 
                + repr(os.listdir(outputfile_dir)) )

    def save_config_batch(self, file_name):
        name_output_file = self.var_configName.get()#( 'config_Nov_auto.ini' )
        batch_name = self.var_batchName.get()
        outputfile_dir = os.path.join(configs_dir, batch_name)
        file_path = os.path.join(configs_dir,self.var_batchName.get(), 
                                 name_output_file+"_s"+repr(seed)+".ini")
        if not os.path.exists(outputfile_dir):
            os.makedirs(outputfile_dir)
        text_file = open(file_path, "w")
        text_file.write(self.config)
        text_file.close()

    def createRunBatchButton(self):
        B_run_batch = tk.Button(self.page5, text="Run batch")
        B_run_batch['command'] = self.response_run_batch
        B_run_batch.grid(sticky=tk.N+tk.S+tk.W+tk.E)
        
    def response_run_batch(self):
        start_time = time.time() # start clock
        
        batch_name = self.var_batchName.get()
        batch_dir = os.path.join(configs_dir, batch_name)
        jobs_list = os.listdir(batch_dir)
        results = {}
        results_table = pd.DataFrame(data=np.nan, index=range(len(jobs_list)),
            columns=['job','seed','ntrain','factor','nnoise']+self.ADmethods)
        
        for i in range(len(jobs_list)):
            # name of job
            job = jobs_list[i]
            results[job] = execute_job(batch_name+"/"+job)
            results_table.loc[i,'job'] = job
                             
            # seed for data split
            job_config_data = configparser.ConfigParser()
            job_config_data.read(os.path.join(batch_dir, job))  
            results_table.loc[i,'seed'] = ast.literal_eval(
                    job_config_data['Data']['params'])['seed']
            
            # number of training samples 
            results_table.loc[i,'ntrain'] = ast.literal_eval(
                    job_config_data['Data']['params'])['n_train_samples_max']
            
            # representation factor 
            results_table.loc[i,'factor'] = ast.literal_eval(
                    job_config_data['Rep1']['params'])['factor']
            
            # number of noise features
            results_table.loc[i,'nnoise'] = ast.literal_eval(
                    job_config_data['Data']['params'])['n_noise_cols']
            
            # performances
            for model in results[job]:
                results_table.loc[i,model.ID] = model.performance
                                 
            print ( '%%% FINISHED JOB ' + repr(i+1) 
                    + ' / '+ repr(len(jobs_list)) + ' IN BATCH %%%\n\n' )
                                 
        results_table.to_excel(os.path.join(
                home_dir,'results','results_'+batch_name+'.xlsx') )
        
        time_elapsed = round( time.time() - start_time, 1 ) # stop clock

        print '%%% Batch Run Finished! %%%'
        print '%%% results saved in file results_'+batch_name+'.xlsx %%%'
        print '%%% time elapsed: ' + repr(time_elapsed) + 's %%%'
        
                                         
    def createRunButton(self):
        self.run_standby_text = "Run!"
        self.B_run = tk.Button(self.page5, text=self.run_standby_text)
        self.B_run['command'] = self.response_run
        self.B_run.grid(sticky=tk.N+tk.S+tk.W+tk.E)
        
        self.label_dataset = tk.Label(self.master)
        self.label_dataset.grid()


    def response_run(self):
        ### if checks passed without error, execute test
        self.executeTest()


    def check_valid(self):
        ### check if a dataset has been selected
        if self.dataset_sel is None:
            tkMessageBox.showinfo( "Choose a dataset", 
                                   "Please select a dataset." )
            return; # to stop function in this case
        
        ### check if at least one AD algorithm has been selected
        one_model_sel = False
        for method in self.ADmethods_sel.keys():
            if self.ADmethods_sel[method].get() != 0:
                one_model_sel = True
        if one_model_sel is False:
            tkMessageBox.showinfo( "Choose an AD model", 
                                   ("Please select at least one " + 
                                    "anomaly detection model.") )
            return; # to stop function in this case
        
    
    def executeTest(self):
        self.WriteSave_config()
        self.results = execute_job(self.var_configName.get()+'.ini') 
        self.createResultsWindow(self.results)


    def WriteSave_config(self): ## CHANGE NAME !!!!
        self.write_config()
        
        name_output_file = self.var_configName.get()+'.ini'
        outputfile_dir = os.path.join(configs_dir, name_output_file)
        text_file = open(outputfile_dir, "w")
        text_file.write(self.config)
        text_file.close()
    
    
    def write_config(self, seed_data=None):
        self.check_valid()
        
        if seed_data is None:
            seed_data = self.var_seed_data.get()
        else:
            seed_data = repr(seed_data)
        
        if self.var_desc.get() != '':
            description = "# Description: " + self.var_desc.get() + "\n"
        else:
            description = ""
            
        self.config = (
        "###################################################################" +
        "###\n" +
        "### Anomaly Detection configuration file (automatically generated) " +
        "###\n" +
        "###################################################################" +
        "###\n" +
        description + "\n" + 
        "### Dataset ###\n"
        "###############\n"
        "[Data]\n" +
        "source = original\n" +
        "dataset = " + repr(self.dataset_sel) + "\n" +
        "mode = " + repr("Novelty" if self.var_mode.get() == 0 else "Outlier") 
        + "\n" +
        "params = {'n_train_samples_max': " + self.S_ntrain_max.get() + ", " + 
        "'anom_ratio': " + self.E_ratio.get() + ", " +
        "'n_test_samples': "+ self.S_ntest.get() + ", " +
        "'seed': "+ seed_data + ", " + 
        "'values_norm': " + self.var_values_norm.get() + ", " +
        "'values_anom': " + self.var_values_anom.get() + ", " +
        "'n_noise_cols': " + self.var_nnoise.get() + "}\n" +
        "print_overview = True\n" +
        "visualize_test = False\n\n" +
        "### Representation ###\n" +
        "######################\n" +
        "[Representation]\n" + 
        "n_models = " + self.var_nReps.get() + "\n\n" )

        
        for i in range(int(self.var_nReps.get())):
            self.config = ( self.config +
            "[Rep" + repr(i+1) + "]\n" +
            "model_type = "+ repr(self.RepModels_sel[i].get()) + "\n" + 
            "params = {'factor':" + self.E_Reps_factor[i].get() + "}" + "\n" )
            
            if self.E_Reps[i].get() != "":
                self.config = (self.config +
                "params = "+ (self.E_Reps[i].get()) )
                
            self.config = self.config + "\n\n"
        
        n_models = 0
        for method in self.ADmethods_sel.keys():
            if self.ADmethods_sel[method].get() != 0:
                n_models +=1 
        
        self.config = (self.config +
        "### "+repr("Novelty" if self.var_mode.get() == 0 else "Outlier")
        +" detection methods ###\n" +
        "#################################\n" +
        "[AnomalyMethods]\n" +
        "n_models = " + repr(n_models) + "\n" + 
        "optimize = False\n\n" )

        i = 1
        for method in self.ADmethods_sel.keys():
            if self.ADmethods_sel[method].get() != 0:
                self.config = ( self.config + 
                "[Anom" + repr(i)+"]\n" +
                "model_type = " + repr(method) + "\n" )
                
                self.config = (self.config + "params = {")
                for param in self.ADparams_values[method]:
                    self.config = (self.config + 
                                   repr(param) + " : " + 
                                   self.ADparams_values[method][param].get() 
                                   + ", " )
                self.config = self.config + "}"
                
                i += 1
                self.config += "\n"
        
        
    def createResultsWindow(self, results):
        self.newWindow = tk.Toplevel(self.master)
        self.result_app = Results_App(self.newWindow, results)
        
    def createContentPage6(self):
        L = tk.Label(self.page6, text = "Page5")
        L.grid(row=6, sticky=tk.W)
        

        ### INSERT A PICTURE LIKE THIS !!!!
        #self.image = Image.open("picture.jpg")
        #self.photo = ImageTk.PhotoImage(self.image)
        #self.label = tk.Label(self.page6, image=self.photo)
        #self.label.grid()

###############################################################################
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###############################################################################

class Results_App:
    def __init__(self, master, results):
        self.master = master
        self.results = results
        self.F_master = tk.Frame(self.master, width=500, height=200)
        self.F_master.pack(fill=tk.BOTH, expand=1)
        self.createWidgets()
    
    
    def createWidgets(self):
        self.createPerfWidget()
        
        
    def createPerfWidget(self):
        self.LF_Perf = tk.LabelFrame(self.F_master, text="Performances")
        self.LF_Perf.pack(fill=tk.BOTH, expand=1)
        
        row = 0
        self.L_modeltype = tk.Label(self.LF_Perf, text="MODEL")
        self.L_modeltype.grid(row = row, column = 0, sticky = tk.W)
        self.L_perf = tk.Label(self.LF_Perf, text="AUC - PERFORMANCE")
        self.L_perf.grid(row = row, column = 1, sticky = tk.W)
        self.L_fittime = tk.Label(self.LF_Perf, text="FIT TIME (in s)")
        self.L_fittime.grid(row = row, column = 2, sticky = tk.W)
        self.L_scoretime = tk.Label(self.LF_Perf, text="SCORE TIME (in s)")
        self.L_scoretime.grid(row = row, column = 3, sticky = tk.W)
        
        row +=1

        self.L_perfs_models = {}
        self.L_perfs_AUC = {}
        self.L_perfs_fittime = {}
        self.L_perfs_scoretime = {}
        
        
        self.DF_output = OutputDF(self.results)
        print self.DF_output
        
        for i in self.DF_output.index:
            AUC        = self.DF_output['AUC'][i]
            model      = self.DF_output['model'][i]
            fit_time        = self.DF_output['fit_time'][i]
            score_time      = self.DF_output['score_time'][i]
            AUC_col         = self.DF_output['AUC_col'][i]
            fit_time_col    = self.DF_output['fit_time_col'][i]
            score_time_col  = self.DF_output['score_time_col'][i]
            
            self.L_perfs_models[model] = tk.Label(self.LF_Perf, text = model )
            self.L_perfs_AUC[model] = tk.Label(
                    self.LF_Perf, text=AUC, bg=AUC_col)
            self.L_perfs_fittime[model] = tk.Label(
                    self.LF_Perf, text=fit_time, bg=fit_time_col)
            self.L_perfs_scoretime[model] = tk.Label(
                    self.LF_Perf, text=score_time, bg=score_time_col)
            
            
            self.L_perfs_models[model].grid(row = row, column=0, sticky=tk.W)
            self.L_perfs_AUC[model].grid(row = row, column=1, sticky=tk.E)
            self.L_perfs_fittime[model].grid(row = row, column=2, sticky=tk.E)
            self.L_perfs_scoretime[model].grid(row = row, column=3, 
                                               sticky=tk.E)
            row +=1

###############################################################################
###############################################################################
###############################################################################



###############################################################################
###############################################################################
###############################################################################

class Overview_App:
    
    def __init__(self, master, dataset):
        self.master = master
        self.dataset = dataset

        myAnomalyDataSet = AnomalyDataSet(dataset = dataset, mode = "Novelty")
        print myAnomalyDataSet.dataset
        myAnomalyDataSet.set_params_by_config(config_data)
        myAnomalyDataSet.load_data_from_folder(home_dir)
        self.overview = myAnomalyDataSet.get_overview()
        
        tex = tk.Text(master=master)
        tex.insert(tk.END, self.overview)
        tex.see(tk.END)     
        tex.pack(side=tk.RIGHT)
        
###############################################################################
###############################################################################
###############################################################################



###############################################################################

def OutputDF(results):
    DF_results = pd.DataFrame(
            data = None, 
            index=range(len(results)), 
            columns=['model','AUC','AUC_col','fit_time','fit_time_col',
                     'score_time','score_time_col'] )
                    
    row = 0
    for model in results:
        DF_results.loc[row,] = [
                model.model_type, 
                round(model.performance,4), None,
                model.fit_time, None,
                model.score_time, None ]
        row +=1
    
    cols_num = ['AUC','fit_time','score_time']
    cols_str = ['model','AUC_col','fit_time_col','score_time_col']
    DF_results[cols_num] = DF_results[cols_num].astype(float)
    DF_results[cols_str] = DF_results[cols_str].astype(str)
        
    DF_results['AUC_col']=grad_hex_colors(DF_results['AUC'])
    DF_results['fit_time_col']=grad_hex_colors_inv(DF_results['fit_time'])
    DF_results['score_time_col']=grad_hex_colors_inv(DF_results['score_time'])
    return DF_results;

###############################################################################    


def main():
    root = tk.Tk()
    root.title("Anomaly Detection Test Program")
    root.geometry("950x700")
    app = Settings_App2(root)
    root.mainloop()
    
    
if __name__ == '__main__':
    main()