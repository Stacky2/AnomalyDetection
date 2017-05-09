# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:06:47 2017

@author: Mathias
"""



import sys
import os 
from numpy import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### define directories
Visualization_dir           = os.path.dirname(__file__)
home_dir                    = os.path.normpath(Visualization_dir 
                                           + os.sep + os.pardir)
data_dir                    = os.path.join(home_dir, "data")
AnomalyModels_dir           = os.path.join(home_dir, "AnomalyModels")

### import own modules
## import the AnomalyModel classes
sys.path.insert(0, AnomalyModels_dir)
from AnomalyModels import ( IF_AnomalyModel,   URF_AnomalyModel, 
                            UXGB_AnomalyModel, KMD_AnomalyModel, 
                            KMC_AnomalyModel,  AE_AnomalyModel,
                            DAE_AnomalyModel,  OSVM_AnomalyModel,
                            LSAD_AnomalyModel, FRaC_AnomalyModel,
                            GMM_AnomalyModel,  PCA_AnomalyModel )



### plottingfunction    
def plot_2D_AM_contour(X, model, grid_size=100, nlevels=15, cdf_score=True, 
                       name=None):
    
    model.fit(X)
    scores_pred = model.get_anomaly_scores(X)    
    
    if type(scores_pred) == pd.core.frame.DataFrame:
        scores_pred = np.array(scores_pred.apply(np.mean, axis=1))
    
    #outliers_fraction = 0.1
    #threshold = stats.scoreatpercentile(scores_pred,
    #                                            100 * outliers_fraction)
    factor = 0.5
    
    X_loc = X.copy()
    X_loc.columns = ['x','y']
    x_min, x_max = min(X_loc['x']), max(X_loc['x'])
    x_span = x_max-x_min
    x_left, x_right = x_min - factor*x_span, x_max+factor*x_span
    
    y_min, y_max = min(X_loc['y']), max(X_loc['y'])
    y_span = y_max-y_min
    y_left, y_right = y_min - factor*y_span, y_max+factor*y_span
    
    
     
    xx, yy = np.meshgrid( np.linspace(x_left, x_right, grid_size), 
                          np.linspace(y_left, y_right, grid_size)  )
    temp = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    temp.columns = X.columns
    Z = model.get_anomaly_scores(temp, cdf_score=cdf_score)
    
    if type(Z) == pd.core.frame.DataFrame:
        Z = np.array(Z.apply(np.mean, axis=1))
        
    Z = Z.reshape(xx.shape)
    
    
    plt.figure(figsize=(15, 10)) 
    #plt.title(model.model_type)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), nlevels),
                    cmap=plt.cm.Blues_r)
    
    b1 = plt.scatter(X_loc['x'],X_loc['y'], s=20, c=scores_pred)
    #plt.axis('tight')
    plt.axis('off')
    plt.xlim((x_left, x_right))
    plt.ylim((y_left, y_right))
    #plt.legend([b1],
    #           ["training observations"],
    #           loc="upper left")
    
    if name is None:
        plt.show()
    else:
        plt.savefig(name+'.png')
    
    
"""
### generation of dataset
size = 1000

mean1 = [0,0]
cov1 = [[1,0.8],[0.8,1]]
size1 = size

mean2 = [-2,0]
cov2 = [[1,-0.95],[-0.95,1]]
size2=size

random.seed(1)
X = pd.concat( [ pd.DataFrame(random.multivariate_normal(mean1,cov1,size=size1)),
               pd.DataFrame(random.multivariate_normal(mean2,cov2,size=size2))  ] )
X.columns = ['x','y']



model = PCA_AnomalyModel(mode='Novelty')
plot_2D_AM_contour(X,model,grid_size=100)

model = KMD_AnomalyModel(mode='Novelty', k=10, colsample=1.0)
plot_2D_AM_contour(X,model,grid_size=100)

model = GMM_AnomalyModel(mode='Novelty')
plot_2D_AM_contour(X,model,grid_size=100)


model = IF_AnomalyModel(mode='Novelty', n_estimators=1000)
plot_2D_AM_contour(X,model,grid_size=10, nlevels=15)


"""