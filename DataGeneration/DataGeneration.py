# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:57:15 2016

@author: Mathias

File that contains the functions for data generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import configparser
import os
import ast

from scipy.stats import norm
from operator import add
from itertools import chain
from sklearn import datasets

from multiprocessing import Pool

### dataset config parser
config = configparser.ConfigParser()
config.read( os.path.join(os.path.dirname(__file__), "config.ini"))

Iris_pars = config['Iris']
FCT_pars = config['Forest Cover Type']
CCF_pars = config['Credit Card Fraud']


### define directories
DataGeneration_dir  = os.path.dirname(__file__)
home_dir                = os.path.normpath(DataGeneration_dir 
                                           + os.sep + os.pardir)
data_dir       = os.path.join(home_dir, "data")

### data config parser
config_data = configparser.ConfigParser()
config_data.read( os.path.join(data_dir, "config.ini"))



###############################################################################
### extract data model (only numerical variables) #############################
###############################################################################

def GaussWeights(center, npoints, bw):
    weights = [np.exp( - ( pt - center  )**2 / bw ) for pt in range(1,(npoints+1)) ]
    normalized_weights = [wght/sum(weights) for wght in weights]
    return normalized_weights;
    
def SmoothHistogram(hist, bw):
    counts = hist[0]

    smoothed_counts = np.repeat(np.nan, len(counts))
    
    for i in range(len(counts)):
        smoothed_counts[i] = sum(GaussWeights(i+1, len(counts), bw=bw) * counts)

    sm_hist = list(hist)
    sm_hist[0] = smoothed_counts
    sm_hist = tuple(sm_hist)
    
    return sm_hist;

""" ### for DataModel
def gen_hist_samples(hists, n_samples_hist=-1, seed=1):
    
    ### if number of samples per hist not given take number of points used to
    ### compute histograms
    if n_samples_hist <= 0:
        n_samples = np.sum(hists[0][0])
    
    print n_samples_hist ### TESTING ONLY !!!!
    
    ### initialize histogram samples
    hist_samples = pd.DataFrame( np.nan, 
                                 index   = range(n_samples_hist), 
                                 columns = range(len(hists))      )
    
    ### for each feature generate samples according to the histograms
    for i in range(len(hists)):
        counts = StratWeightedChoice( weights = hists[i][0]/float(np.sum(hists[i][0])), nsamples=n_samples_hist ) #hists[i][0] 
        breaks = hists[i][1]
        
        samples = np.zeros(np.sum(counts)) # ev replace with empty
        temp = 0
        np.random.seed(seed)
        
        ### generate samples for each bin in the histogram
        for j in range(len(counts)):
            samples[temp:(temp+counts[j])] = np.random.uniform( low  = breaks[j], 
                                                   high = breaks[j+1], 
                                                   size = counts[j] )
            temp += counts[j]
            
        hist_samples[i] = samples
        
    return hist_samples;

"""
def gen_hist_samples(hists, n_samples_hist=-1, seed=1): ### for MixedDataModel
    
    ### if number of samples per hist not given take number of points used to
    ### compute histograms
    if n_samples_hist <= 0:
        n_samples_hist = np.sum(hists[0][0])
    
    #print n_samples_hist ### TESTING ONLY !!!!
    
    ### initialize histogram samples
    hist_samples = pd.DataFrame( np.nan, 
                                 index   = range(n_samples_hist), 
                                 columns = range(len(hists))      )
    
    ### for each feature generate samples according to the histograms
    for key in hists.keys():
        counts = StratWeightedChoice( 
                    weights = hists[key][0]/float(np.sum(hists[key][0])), 
                    nsamples=n_samples_hist ) #hists[i][0] 
        breaks = hists[key][1]
        
        samples = np.zeros(np.sum(counts)) # ev replace with empty
        temp = 0
        np.random.seed(seed)
        
        ### generate samples for each bin in the histogram
        for j in range(len(counts)):
            samples[temp:(temp+counts[j])] = np.random.uniform( 
                                                low  = breaks[j], 
                                                high = breaks[j+1], 
                                                size = counts[j]    )
            temp += counts[j]
            
        hist_samples[key] = samples
        
    return hist_samples;




def SmoothPercentile(hist):
    
    ### functions
    def generate_bin_samples(hist):
        counts = hist[0]
        breaks = hist[1]
        midpts = breaks[0:(len(breaks)-1)] + ( breaks[1:len(breaks)] - breaks[0:(len(breaks)-1)]) /2
                                   
        samples = np.empty([0])
        for i in range(len(midpts)):
            samples = np.concatenate( [ samples, 
                                        np.repeat(midpts[i], counts[i]) ] )
        return samples;
        
    
    def bw(x):
        "calculation of 'bw.nrd0' from R"
        def IQR(x):
            "inter quartile range"
            q75, q25 = np.percentile(x, [75 ,25])
            iqr = q75 - q25
            return(iqr)
        
        hi = np.std(x)
        lo = min(hi,IQR(x)/1.34) # if doesnt work look at R implementation of "bw.nrd0"
        bw = 0.9 * lo * len(x)**(-0.2)
        return bw;

        
    def KDE(x):
        t = np.mean( norm.cdf( (x-sample)/bw ) )
        return t;
    vKDE = np.vectorize(KDE)
        
    
    def kde(x):
        t = np.mean( norm.pdf( (x-sample)/bw )/bw )
        return t;
    vkde = np.vectorize(kde)
        
    
    def QKDE(p):
        
        eps = 1e-5 ####################################################################################
        p = max(min(p, 1-eps), eps)
        
        def tempf(t):
            return (vKDE(t)-p);
            
        root = brenth(tempf, Interval[0], Interval[1])
        return root;
    vQKDE = np.vectorize(QKDE)
    
    ### generate the samples (at bin midpoints), estimate bandwidth for density
    ### estimate and boundaries of histogram
    sample = generate_bin_samples(hist)
    bw = bw(sample)
    size_bin = hist[1][1] - hist[1][0]
    tail_size = 2 * size_bin ### MAKE ADJUSTABLE !!!! ######################################################################
    Interval = [min(hist[1])-tail_size , max(hist[1])+tail_size]
    
    ### Plot histogram and density
    ## histogram
    plt.bar(range(len(hist[0])), hist[0] ) 
    plt.show()
    
    ## density
    x = np.linspace(Interval[0],Interval[1],50)
    y=vkde(x)
    plt.plot(x,y)
    plt.show()
    
    ## distribution function
    x = np.linspace(Interval[0],Interval[1],50)
    y=vKDE(x)
    plt.plot(x,y)
    plt.show()
    
    
    return vQKDE;


"""
class DataModel:
    def __init__( self, data, nbins=20 ):
        self.cor   = np.corrcoef(data,rowvar=0)
        self.hists = [] 
        
        for col in data:
            self.hists.append(np.histogram(data[col], bins=nbins))
    
    def generateData(self, nsamples, seed=1)  :
        
        np.random.seed(seed)
        mean = np.repeat(0,len(self.hists))
        T = pd.DataFrame( np.random.multivariate_normal( mean = mean, 
                                                        cov  = self.cor, 
                                                        size = nsamples) )
        U = pd.DataFrame( norm.cdf(T) )
        
        hist_samples = gen_hist_samples(self.hists, seed=seed)
        
        gen_samples = pd.DataFrame( np.nan, 
                                    index   = range(nsamples), 
                                    columns = range(len(self.hists))      )  
        
        for i in range(len(self.hists)):
            gen_samples[i] = np.percentile(a=hist_samples[i], q=100*U[i])
            
        return gen_samples;
"""
###############################################################################

###############################################################################
### Stratified Weighted Choice ################################################
###############################################################################
#for DataModel
def StratWeightedChoice(weights, nsamples, seed=1):
    nsamples_modes =  [ int(nsamples*freq) for freq in weights]
    nsamples_left = nsamples - sum(nsamples_modes)
    np.random.seed(seed)
    samples_left = list(chain( *np.random.multinomial( nsamples_left, weights, 1) ) )
    nsamples_modes = map( add, nsamples_modes, samples_left )
    
    return nsamples_modes;

#for MixedDataModel (weights as dict)
def StratWeightedChoiceDict(weights, nsamples, seed=1):
    nsamples_modes = {}
    for key in weights.keys():
        nsamples_modes[key] = int(nsamples*weights[key])
    
    nsamples_left = nsamples - np.sum(nsamples_modes.values())
    np.random.seed(seed)
    #samples_left = list(chain( *np.random.multinomial( nsamples_left, weights, 1) ) )
    samples_left = np.random.choice(a=weights.keys(), size=nsamples_left, p=weights.values())
    for sample in samples_left:
        nsamples_modes[sample] +=1
    
    return nsamples_modes;
 
###############################################################################


###############################################################################
### extract mode-specifig data model (only numerical variables) ###############
###############################################################################

class DataModel:
    def __init__( self, data, modes=None, smooth_hists=False, 
                 bw=1, nbins=10 ):
        self.cor   = []
        self.hists = [] 
        self.modes = None  # thing if need to save complete vector
        self.freq_modes = None
        
        
        if modes is None:
            self.modes=pd.Categorical(np.repeat(0,data.shape[0]))
        else:
            self.modes=pd.Categorical(modes)
            
        self.freq_modes=list(self.modes.describe().freqs)
        
        for cat in self.modes.categories:
            self.cor.append( np.corrcoef( data[modes==cat], rowvar=0) )
    
            hists = []
            for col in data:
                hist = np.histogram(data.loc[modes==cat,col], bins=nbins)
                if smooth_hists:
                    sm_hist = SmoothHistogram(hist, bw=bw)
                    hists.append(sm_hist)
                else:
                    hists.append(hist)
                
            
            self.hists.append( hists )
    
    def generateData(self, nsamples, method='histogram', seed=1)  :
        
        nsamples_modes = StratWeightedChoice( self.freq_modes, nsamples, 
                                               seed=seed )
        
        
        gen_samples = pd.DataFrame( np.nan, 
                                     index   = range(0), 
                                     columns = range(len(self.hists[0])) )
        """ pd.DataFrame( np.nan, 
                                     index   = range(nsamples), 
                                     columns = range(len(self.hists[0])) ) """
        
        np.random.seed(seed)                
             
        for i in range(len(self.hists)):
            mean = np.repeat(0,len(self.hists[i]))
            T = pd.DataFrame( np.random.multivariate_normal( 
                                 mean = mean, 
                                 cov  = self.cor[i], 
                                 size = nsamples_modes[i])    )
            U = pd.DataFrame( norm.cdf(T) )
            
            gen_samples_mode = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[i]), 
                                        columns = range(len(self.hists[i])) )
            
            if method == 'histogram':
                hist_samples = gen_hist_samples(self.hists[i], n_samples_hist = nsamples, seed=seed)
                for j in range(len(self.hists[i])):
                    gen_samples_mode[j] = np.percentile(a=hist_samples[j], q=100*U[j])
                    
            if method == 'smooth':
                for j in range(len(self.hists[i])):
                    Percentile = SmoothPercentile(self.hists[i][j])
                    print repr(min(U[j])) + ' ' + repr(max(U[j])) ### TESTING ONLY !!!!
                    print repr( Percentile( 100* min(U[j])) ) + ' ' + repr( Percentile( 100* max(U[j])) ) ### TESTING ONLY !!!!
                    gen_samples_mode[j] = Percentile( 100*U[j] )
                    
            
            gen_samples = pd.concat([gen_samples, gen_samples_mode])
                        
            print 'end mode' + repr(i)
            # print repr( gen_samples[ sum(nsamples_modes[:i]) : sum(nsamples_modes[:(i+1)]) ] )
            # print repr( gen_samples_mode )
            # print repr( sum(nsamples_modes[:i]) ) + ' '+ repr( sum(nsamples_modes[:(i+1)]) )

            
        gen_samples.index = range(gen_samples.shape[0])
        return gen_samples;
            

""" BACKUP !!!
    def generateData(self, nsamples, seed=1)  :
        
        nsamples_modes = StratWeightedChoice( self.freq_modes, nsamples, 
                                               seed=seed )
        
        
        gen_samples = pd.DataFrame( np.nan, 
                                     index   = range(0), 
                                     columns = range(len(self.hists[0])) )
        #"" pd.DataFrame( np.nan, 
        #                              index   = range(nsamples), 
        #                             columns = range(len(self.hists[0])) ) ""
        
        np.random.seed(seed)                
             
        for i in range(len(self.hists)):
            mean = np.repeat(0,len(self.hists[i]))
            T = pd.DataFrame( np.random.multivariate_normal( mean = mean, 
                                                             cov  = self.cor[i], 
                                                             size = nsamples_modes[i]) )
            U = pd.DataFrame( norm.cdf(T) )
            
            hist_samples = gen_hist_samples(self.hists[i], n_samples_hist = nsamples, seed=seed)
            
            gen_samples_mode = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[i]), 
                                        columns = range(len(self.hists[i])) )
            for j in range(len(self.hists[i])):
                gen_samples_mode[j] = np.percentile(a=hist_samples[j], q=100*U[j])
            
            # gen_samples[ sum(nsamples_modes[:i]) : sum(nsamples_modes[:(i+1)]) ] = gen_samples_mode
            
            gen_samples = pd.concat([gen_samples, gen_samples_mode])
                        
            print 'end mode' + repr(i)
            # print repr( gen_samples[ sum(nsamples_modes[:i]) : sum(nsamples_modes[:(i+1)]) ] )
            # print repr( gen_samples_mode )
            # print repr( sum(nsamples_modes[:i]) ) + ' '+ repr( sum(nsamples_modes[:(i+1)]) )

            
        gen_samples.index = range(gen_samples.shape[0])
        return gen_samples;
"""

###############################################################################


###############################################################################
### extract mode-specifig data model (numerical and categorical variables #####
###############################################################################

class MixedDataModel:
    def __init__( self, data, modes=None, smooth_hists=False, 
                 bw=1, nbins=10 ):
        self.cor        = {}
        self.cols       = None
        self.dum_cols   = None
        self.num_cols   = []
        self.cat_cols   = {}
        self.cats       = {}
        self.empty_cols = {}
        self.hists      = {}
        self.probs      = {}
        self.modes      = None  # thing if need to save complete vector
        self.freq_modes = None
        
        ### create dummy version of dataset
        data_dummy = pd.get_dummies(data)
        
        ###
        self.cols = data.columns
        self.dum_cols = data_dummy.columns
        for col in data:
            if data[col].dtype.name == 'category':
                new_cols = [ str(col) + '_' + str(level)\
                             for level in\
                             pd.Categorical( data[col] ).categories.values ]
                self.cat_cols[col] = new_cols
                self.cats[col] = pd.Categorical( data[col] ).categories.values
                self.probs[col] = pd.Categorical(data[col]).describe().freqs.to_dict()
            else:
                self.num_cols.append(col)
        
        ### define mode related variables
        if modes is None:
            modes=pd.Categorical(np.repeat(0,data_dummy.shape[0]))
        else:
            modes=pd.Categorical(modes)
            
        self.modes = modes.categories.values
        self.freq_modes=modes.describe().freqs.to_dict()
        
        
        for cat in modes.categories:
            self.cor[cat] = pd.DataFrame( np.corrcoef( data_dummy[modes==cat], rowvar=0 ) )
            
            self.cor[cat] = self.cor[cat].fillna(value=0.0)
            self.empty_cols[cat] = self.dum_cols[self.cor[cat].isnull().all(1)]##########################################################
    
            hists = {}
            for col in data_dummy:
                
                if col in self.num_cols:
                    hist = np.histogram(data_dummy.loc[modes==cat,col], bins=nbins)
                    if smooth_hists:
                        sm_hist = SmoothHistogram(hist, bw=bw)
                        hists[col] = sm_hist
                    else:
                        hists[col] = hist
                
            
            self.hists[cat] = hists
    
    def generateData(self, nsamples, method='histogram', seed=1) :
        
        nsamples_modes = StratWeightedChoiceDict( self.freq_modes, nsamples, 
                                               seed=seed )
        
        gen_samples = pd.DataFrame( np.nan, 
                                     index   = range(0), 
                                     columns = self.cols )
        gen_modes = np.array([])
        """ pd.DataFrame( np.nan, 
                                     index   = range(nsamples), 
                                     columns = range(len(self.hists[0])) ) """
        
        np.random.seed(seed)                
             
        for key in self.cor.keys():
            mean = np.repeat(0,self.cor[key].shape[0])
            T = pd.DataFrame( np.random.multivariate_normal( mean = mean, 
                                                             cov  = self.cor[key], 
                                                             size = nsamples_modes[key]) )
            U = pd.DataFrame( norm.cdf(T) )
            U.columns = self.dum_cols
            
            gen_samples_mode_num = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[key]), 
                                        columns = [] ) #range(self.cor[i].shape[0]) )
            gen_samples_mode = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[key]), 
                                        columns = [] ) #range(self.cor[i].shape[0]) )
            
            print 'generate histogram samples...'
            if method == 'histogram':
                hist_samples = gen_hist_samples(self.hists[key], n_samples_hist = nsamples, seed=seed)
                for key2 in self.hists[key].keys() :
                    gen_samples_mode_num[key2] = np.percentile(a=hist_samples[key2], q=100*U[key2])
                    
            if method == 'smooth':
                """
                for j in range(len(self.hists[i])):
                    Percentile = SmoothPercentile(self.hists[i][j])
                    print repr(min(U[j])) + ' ' + repr(max(U[j])) ### TESTING ONLY !!!!
                    print repr( Percentile( 100* min(U[j])) ) + ' ' + repr( Percentile( 100* max(U[j])) ) ### TESTING ONLY !!!!
                    gen_samples_mode[j] = Percentile( 100*U[j] )
                """
            
            for col in self.cols:
                if col in self.num_cols:
                    gen_samples_mode[col] = gen_samples_mode_num[col]
                else:
                    cats=self.cats[col]
                    cat_cols = self.cat_cols[col]
                    level_probs = self.probs[col]
                    
                    def sample_category(x):
                        x = np.multiply( x, np.array(level_probs.values()) )
                        y =  np.random.choice(a=cats, size=1, p=x/np.sum(x))[0] #np.random.choice(a=cats, size=1, p=x/np.sum(x))
                        return y;
                        
                    vsample_category = np.vectorize(sample_category, otypes=[pd.Categorical]) # NEW!!!
                        
                    cats_unif = U[cat_cols]
                    
                    
                    #cats_sampled = pd.Categorical(cats_unif.apply(sample_category, axis=1))# OLD!!!
                    cats_unif_arr = np.array(cats_unif) # NEW!!!
                    cats_sampled = vsample_category(cats_unif_arr) # NEW!!!
                    
                    gen_samples_mode[col] = cats_sampled
            
            gen_samples = pd.concat([gen_samples, gen_samples_mode])
            gen_modes = np.concatenate( 
                           [ gen_modes,
                             np.repeat(key, gen_samples_mode.shape[0]) ] )
                        
            print 'end mode' + repr(key)

            
        gen_samples.index = range(gen_samples.shape[0])
        return gen_samples, gen_modes;
        
        
    def generateDataModes(self, nsamples_modes, method='histogram', seed=1) :
        
        gen_samples = pd.DataFrame( np.nan, 
                                     index   = range(0), 
                                     columns = self.cols )
        gen_modes = np.array([])
        """ pd.DataFrame( np.nan, 
                                     index   = range(nsamples), 
                                     columns = range(len(self.hists[0])) ) """
        
        np.random.seed(seed)                
             
        for key in nsamples_modes.keys():
            mean = np.repeat(0,self.cor[key].shape[0])
            T = pd.DataFrame( np.random.multivariate_normal( mean = mean, 
                                                             cov  = self.cor[key], 
                                                             size = nsamples_modes[key]) )
            U = pd.DataFrame( norm.cdf(T) )
            U.columns = self.dum_cols
            
            gen_samples_mode_num = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[key]), 
                                        columns = [] ) #range(self.cor[i].shape[0]) )
            gen_samples_mode = pd.DataFrame( np.nan, 
                                        index   = range(nsamples_modes[key]), 
                                        columns = [] ) #range(self.cor[i].shape[0]) )
            
            print 'generate histogram samples...'
            if method == 'histogram':
                hist_samples = gen_hist_samples(self.hists[key], n_samples_hist = np.sum(nsamples_modes.values()), seed=seed)
                for key2 in self.hists[key].keys() :
                    gen_samples_mode_num[key2] = np.percentile(a=hist_samples[key2], q=100*U[key2])
                    
            if method == 'smooth':
                """
                for j in range(len(self.hists[i])):
                    Percentile = SmoothPercentile(self.hists[i][j])
                    print repr(min(U[j])) + ' ' + repr(max(U[j])) ### TESTING ONLY !!!!
                    print repr( Percentile( 100* min(U[j])) ) + ' ' + repr( Percentile( 100* max(U[j])) ) ### TESTING ONLY !!!!
                    gen_samples_mode[j] = Percentile( 100*U[j] )
                """
            
            for col in self.cols:
                print col
                if col in self.num_cols:
                    gen_samples_mode[col] = gen_samples_mode_num[col]
                else:
                    cats=self.cats[col]
                    cat_cols = self.cat_cols[col]
                    level_probs = self.probs[col]
                    
                    def sample_category(x):
                        x = np.multiply( x, np.array(level_probs.values()) )
                        y =  np.random.choice(a=cats, size=1, p=x/np.sum(x))[0] #np.random.choice(a=cats, size=1, p=x/np.sum(x))
                        return y;
                        

                    vsample_category = np.vectorize(sample_category)
                        
                    cats_unif = U[cat_cols]
                    #cats_unif_arr = np.array(cats_unif) # NEW!!!
                    cats_sampled = pd.Categorical(cats_unif.apply(sample_category, axis=1)) # OLD!!!
                    #cats_sampled = pd.Categorical( vsample_category(cats_unif.as_matrix()) )
                    #print repr(cats_sampled)
                    
                    gen_samples_mode[col] = cats_sampled
            
            gen_samples = pd.concat([gen_samples, gen_samples_mode])
            gen_modes = np.concatenate( 
                           [ gen_modes,
                             np.repeat(key, gen_samples_mode.shape[0]) ] )
                        
            print 'Finished generation of mode' + repr(key)

            
        gen_samples.index = range(gen_samples.shape[0])
        return gen_samples, gen_modes;
        

    def generateNoveltyAnomalyData(self, nsamples_train, nsamples_test, 
                                   anom_ratio, anom_modes,
                                   norm_modes, method='histogram', seed=1):

        ### functions
        def normalize_dict(dict):
            sum = np.sum(dict.values())
            for key in dict.keys():
                dict[key] = dict[key]/sum
            return dict;

        
        ### if only one of anom_modes or norm_modes is given assign the 
        ### complement of self.modes to the other one
        if anom_modes is None:
            anom_modes = [mode for mode in self.modes\
                          if not mode in norm_modes]
        if norm_modes is None:
            norm_modes = [mode for mode in self.modes\
                          if not mode in anom_modes]
        
        ### calculate number of samples in each mode in train and test
        freq_modes_train = {}
        for mode in norm_modes:
            freq_modes_train[mode] = self.freq_modes[mode]
        freq_modes_train = normalize_dict(freq_modes_train)
        nsamples_modes_train = StratWeightedChoiceDict( 
                                  freq_modes_train, 
                                  nsamples_train,
                                  seed=seed )
        #####
        freq_modes_test = {}
        
        for mode in norm_modes:
            freq_modes_test[mode] = ( self.freq_modes[mode] 
                                      * (1.0 / (1+anom_ratio)) 
                                      * (1.0/len(norm_modes)) )
            
        for mode in anom_modes:
            freq_modes_test[mode] = ( self.freq_modes[mode] 
                                      * (anom_ratio / (1.0+anom_ratio)) 
                                      * (1.0/len(anom_modes)) )
            
        freq_modes_test = normalize_dict(freq_modes_test)
        nsamples_modes_test = StratWeightedChoiceDict( 
                                  freq_modes_test, 
                                  nsamples_test,
                                  seed=seed+1 )
        
        #####
        
        ### generate train, test and label_test
        print 'Generate train dataset ...'
        train, label_train = self.generateDataModes( 
                                     nsamples_modes_train, 
                                     method='histogram', seed=1)
        print 'Generate test dataset ...'
        test, label_test = self.generateDataModes(
                                   nsamples_modes_test,
                                   method='histogram', seed=1)
        
        label_test = np.array( [int(label in anom_modes) for label in label_test] )
        
        return train, test, label_test;
        
###############################################################################


###############################################################################
### generate data #############################################################
###############################################################################

def generateNoveltyData( dataset, 
                         nsamples_train    = None, 
                         nsamples_test     = None, 
                         anom_ratio        = None  ):
    
    
    if dataset == 'Iris':
        ### load data
        iris_import = datasets.load_iris()
        modes=pd.Categorical(iris_import.target)
        modes=modes.rename_categories(iris_import.target_names)
        iris = pd.DataFrame(iris_import.data)
        iris['species'] = modes
        del iris['species']
        
        ### fit data model
        iris_model = MixedDataModel( iris, 
                                     modes = modes, 
                                     nbins = int(Iris_pars['nbins']), 
                                     smooth_hists = True, 
                                     bw = float(Iris_pars['bw']) )
        
        ### generate novelty data
        train, test , label_test = iris_model.generateNoveltyAnomalyData(
           nsamples_train  =   int(Iris_pars['nsamples_train']), 
           nsamples_test   =   int(Iris_pars['nsamples_test']), 
           anom_ratio      = float(Iris_pars['anom_ratio']),
           anom_modes      = ast.literal_eval(Iris_pars['anom_modes']), 
           norm_modes      = ast.literal_eval(Iris_pars['norm_modes']), 
           method          = 'histogram', 
           seed            =   int(Iris_pars['seed']) )
        
        return train, test, label_test;
        
    elif dataset == 'Forest Cover Type':
        
        ### load data
        FCT_dir = os.path.join( data_dir, 'Forest Cover Type', 'data.csv' )
        FCT_import = pd.read_csv(FCT_dir)
        label = 'Cover_Type'
        modes = pd.Categorical( FCT_import[label] ) 
        FCT = FCT_import.copy()
        del FCT[label]
        
        fac_cols = ast.literal_eval(config_data['Forest Cover Type']['fac_cols'])
        for col in fac_cols:
            FCT[col] = pd.Categorical(FCT[col])
        
        
        ### fit data model
        FCT_model = MixedDataModel( FCT, 
                                    modes = modes, 
                                    nbins = int(FCT_pars['nbins']), 
                                    smooth_hists = True, 
                                    bw = float(FCT_pars['bw']) )
        
        ### generate novelty data
        train, test , label_test = FCT_model.generateNoveltyAnomalyData(
           nsamples_train  =   int(FCT_pars['nsamples_train']), 
           nsamples_test   =   int(FCT_pars['nsamples_test']), 
           anom_ratio      = float(FCT_pars['anom_ratio']),
           anom_modes      = ast.literal_eval(FCT_pars['anom_modes']), 
           norm_modes      = ast.literal_eval(FCT_pars['norm_modes']), 
           method          = 'histogram', 
           seed            =   int(FCT_pars['seed']) )
        
        return train, test, label_test;
        
    
    elif dataset == 'Credit Card Fraud':
        
        ### load data
        CCF_dir = os.path.join( data_dir, 'Credit Card Fraud', 'data.csv' )
        CCF_import = pd.read_csv(CCF_dir)
        label = 'Class'
        modes = pd.Categorical( CCF_import[label] ) 
        CCF = CCF_import.copy()
        del CCF[label]
        
        fac_cols = ast.literal_eval(config_data['Credit Card Fraud']['fac_cols'])
        for col in fac_cols:
            CCF[col] = pd.Categorical(CCF[col])
        
        
        ### fit data model
        CCF_model = MixedDataModel( CCF, 
                                    modes = modes, 
                                    nbins = int(CCF_pars['nbins']), 
                                    smooth_hists = True, 
                                    bw = float(CCF_pars['bw']) )
        
        ### generate novelty data
        train, test , label_test = CCF_model.generateNoveltyAnomalyData(
           nsamples_train  =   int(CCF_pars['nsamples_train']), 
           nsamples_test   =   int(CCF_pars['nsamples_test']), 
           anom_ratio      = float(CCF_pars['anom_ratio']),
           anom_modes      = ast.literal_eval(CCF_pars['anom_modes']), 
           norm_modes      = ast.literal_eval(CCF_pars['norm_modes']), 
           method          = 'histogram', 
           seed            =   int(CCF_pars['seed']) )
        
        return train, test, label_test;
        
        
    else:
        raise Exception( 'There are no generation settings for dataset "' +
                         dataset + '" implemented.' )
        
###############################################################################


"""
### TESTING ONLY !!!

### Iris ###
############

### load data
iris_import = datasets.load_iris()
modes=pd.Categorical(iris_import.target)
modes=modes.rename_categories(['a','b','c'])
iris = pd.DataFrame(iris_import.data)
iris['species'] = modes
del iris['species']

### fit data model
iris_model = MixedDataModel( iris, modes=modes, 
                             nbins=20, smooth_hists=True, bw=0.5 )

### generate data
nsamples=150
iris_syn, modes_iris = iris_model.generateData( nsamples=nsamples, 
                                                method = 'histogram' )

### plot data
axes = pd.tools.plotting.scatter_matrix(iris, alpha=0.2, figsize=(10, 10))
axes = pd.tools.plotting.scatter_matrix(iris_syn, alpha=0.2, figsize=(10, 10))

### check marginal distributions of categorical variable 'species'
print pd.Categorical( iris['species'] ).describe()
print pd.Categorical( iris_syn['species'] ).describe()

### check difference of correlation matrices
cor_diff = ( iris_model.cor[0] 
             - np.corrcoef( pd.get_dummies(iris_syn), rowvar=0 ) )
np.max(abs(cor_diff))
np.mean(np.array(cor_diff))

### Anomaly samples
bla, bla2 = iris_model.generateDataModes( {'a':1000}, method='histogram', seed=1)
axes = pd.tools.plotting.scatter_matrix(bla, alpha=0.2, figsize=(10, 10))

tra,te,lab_te, = iris_model.generateNoveltyAnomalyData(
              nsamples_train=1000, nsamples_test=1000, anom_ratio=0.1, 
              anom_modes=['b'], norm_modes=None, method='histogram', seed=1)

axes = pd.tools.plotting.scatter_matrix(te, alpha=0.2, figsize=(10, 10))



### Boston housing data ###
###########################

### load and prepare data
from sklearn.datasets import load_boston
boston_import = load_boston()
boston = pd.DataFrame( boston_import.data )
boston.columns = boston_import.feature_names
fac_cols = ['CHAS','RAD']
for col in fac_cols:
    boston[col] = pd.Categorical( boston[col].astype(int) )
label_col = 'RAD'
modes = pd.Categorical( boston[label_col] )
#boston = boston.drop(label_col,1)

### fit data model
boston_model = MixedDataModel( boston, modes=None, 
                               nbins=50, smooth_hists=True, bw=0.5 )
boston_syn, modes_boston = boston_model.generateData( nsamples=506, method = 'histogram' )
    
### plot data
axes = pd.tools.plotting.scatter_matrix(boston, alpha=0.2, figsize=(10, 10))
axes = pd.tools.plotting.scatter_matrix(boston_syn, alpha=0.2, figsize=(10, 10))

### check marginal distributions of categorical variable 'species'
print pd.Categorical( boston['CHAS'] ).describe()
print pd.Categorical( boston_syn['CHAS'] ).describe()
print pd.Categorical( boston['RAD'] ).describe()
print pd.Categorical( boston_syn['RAD'] ).describe()

### check difference of correlation matrices
cor_diff = boston_model.cor[0] - np.corrcoef( pd.get_dummies(boston_syn), rowvar=0 )
np.max(abs(cor_diff))
np.mean(np.array(cor_diff))


### Credit Card Fraud ###
#########################
cor_syn_0 = pd.DataFrame(np.corrcoef(train, rowvar=0))
cor_real_0 = pd.DataFrame(np.corrcoef(CCF[modes=='Class0'], rowvar=0))

np.max(np.max(cor_syn_0-cor_real_0))
np.mean(np.mean(cor_syn_0-cor_real_0))

axes = pd.tools.plotting.scatter_matrix(train.iloc[1:1000,1:10], alpha=0.2, figsize=(10, 10))
axes = pd.tools.plotting.scatter_matrix(CCF.iloc[1:1000,1:10], alpha=0.2, figsize=(10, 10))

plt.hist(train.iloc[:,2], 100)
plt.hist(CCF.iloc[:,2], 100)

plt.hist()
"""