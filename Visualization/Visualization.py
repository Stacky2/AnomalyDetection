# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:24:16 2016

@author: Mathias

This file contains the function for the visualization of the data.
"""

### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

import time

from sklearn.manifold import TSNE


###############################################################################
### TSNE visualization ########################################################
###############################################################################

def plot_samples_TSNE(X, label, nsamples=100, seed=1):
    
    start_time = time.time() # start clock
    
    ### select part of data and label 
    rd.seed(seed)
    X_copy = X.copy()
    X_copy.index = range(X_copy.shape[0])
    IDX_sample = rd.sample( range(X.shape[0]), k=min(nsamples,X.shape[0]) )
    Xsel = pd.get_dummies(X_copy.loc[IDX_sample,:])
    label_sel = label[IDX_sample]
    
    ### fit TSNE model
    model = TSNE(n_components=2, random_state=seed, verbose=1)
    np.set_printoptions(suppress=True)
    X_TSNE = model.fit_transform( Xsel ) 
    
    ### plot TSNE with col as colour label
    plt.figure()
    plt.figure(figsize=(15, 10)) 
    col_list = ['b','g','r','m','y','c','black'] # add more
    for i in range(np.max(label)+1):
        color = col_list[i]
            
        plt.scatter( X_TSNE[np.where(label_sel==i),0], 
                     X_TSNE[np.where(label_sel==i),1], 
                     marker='x', color=color,
                     linewidth='1', alpha=0.8, label=repr(i) )
    
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('T-SNE on ' + repr(nsamples) + ' train samples')
    plt.legend(loc='best')
    # plt.savefig('1.png')
    plt.show()
    
    time_elapsed = round(time.time() - start_time, 1) # stop clock
    print ('  model fit finished! (time elapsed: ' 
            + repr(time_elapsed) + 's)\n')
    
###############################################################################

###############################################################################
### Functions for color gradient according to list of values ##################
###############################################################################

col_low =  "#cc0000"  # "#b20000" #"#7f0000" #"#ff0000"
col_high = "#007f00" # "#006600"
col_white = "#ffffff"

def grad_hex_colors(values, n =10000):
    if len(values) == 1:
        return None;
    values = np.array(values)
    bins = np.linspace(start=min(values), stop = max(values), num=n)
    
    color_grad1 = linear_gradient(col_low,col_white, n = int((n/2.0)+1))['hex']
    color_grad2 = linear_gradient(col_white,col_high, n = int((n/2.0)+1))['hex']
    color_grad = color_grad1 + color_grad2
    
    values_bined = list(np.digitize(values, bins, right=False))
    colors= [color_grad[Bin] for Bin in values_bined ]
    return colors;

def grad_hex_colors_inv(values, n =10000):
    if len(values) == 1:
        return None;
    values = np.array(values)
    bins = np.linspace(start=min(values), stop = max(values), num=n)
    
    color_grad1 = linear_gradient(col_high,col_white, n = int((n/2.0)+1))['hex']
    color_grad2 = linear_gradient(col_white,col_low, n = int((n/2.0)+1))['hex']
    color_grad = color_grad1 + color_grad2
    #color_grad= linear_gradient(col_high,col_low, n =n+1)['hex'] # green:"#00ff00"
    
    values_bined = list(np.digitize(values, bins, right=False))
    colors= [color_grad[Bin] for Bin in values_bined ]
    return colors;


def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
    
    
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

###############################################################################