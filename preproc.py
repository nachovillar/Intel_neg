# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    X = pd.read_csv('x_input.csv', sep=",", header=None)
    Y = pd.read_csv('y_output.csv', sep=",", header=None)
    config = pd.read_csv('config.csv', sep=",", header=None)
    
    #X = X.sample(frac=1)
    #Y = Y.sample(frac=1)
    
    p = config.loc[0, 0]
    nodos_ocultos = config.loc[1, 0]
    penalidad_pinversa = config.loc[2, 0]
    
    a = 0.01
    b = 0.99
    
    X_min = X.min(axis=1)
    X_max = X.max(axis=1)
    
    Y_min = Y.min(axis=1)
    Y_max = Y.max(axis=1)
    
    X = X.T
    Y = Y.T
    
    normalized_X = (X-X_min)/(X_max-X_min)
    normalized_X = (b-a)*normalized_X + a
    normalized_X = normalized_X.T
  
    normalized_Y = (Y-Y_min)/(Y_max-Y_min)
    normalized_Y = (b-a)*normalized_Y + a
    normalized_Y = normalized_Y.T
    
    output_data_X = pd.DataFrame(normalized_X)
    output_data_X[np.isnan(normalized_X)] = 0

    output_data_Y = pd.DataFrame(normalized_Y)
    output_data_Y[np.isnan(normalized_Y)] = 0
    
    output_data_X.to_csv(path_or_buf = 'X.csv', index = False    , mode = 'w+')
    output_data_Y.to_csv(path_or_buf = 'Y.csv', index = False    , mode = 'w+')