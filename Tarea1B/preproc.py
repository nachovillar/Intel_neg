import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    X = pd.read_csv('x_input.csv', sep=",", header=None)
    Y = pd.read_csv('y_output.csv', sep=",", header=None)
    config = pd.read_csv('config.csv', sep=",", header=None)

    p = config.loc[0, 0] /100
    hn = config.loc[1, 0]
    mu = config.loc[2, 0]
    MaxIter = config.loc[3,0]
    
    a = 0.01
    b = 0.99
    
    D, N = X.shape
    L = round(N*p)
    
    
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
    
    xe = normalized_X.iloc[:, 0: L]
    ye = normalized_Y.iloc[:, 0: L]
    
    xv = normalized_X.iloc[:, L:]
    yv = normalized_Y.iloc[:, L:]
    
 
    xe.to_csv(path_or_buf = 'train_x.csv', index = False    , mode = 'w+', header = None)
    ye.to_csv(path_or_buf = 'train_y.csv', index = False    , mode = 'w+', header = None)    
    
    xv.to_csv(path_or_buf = 'test_x.csv', index = False    , mode = 'w+', header = None)
    yv.to_csv(path_or_buf = 'test_y.csv', index = False    , mode = 'w+', header = None)
    