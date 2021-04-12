# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:26:20 2021

@author: pmard
"""
import numpy as np
import pandas as pd
import random

def normalizar(X):
    X=X.T
    X_min = X.min(axis=1)
    X_max = X.max(axis=1)
    aux = (((X-X_min)/(X_max-X_min))*(b-a))+a
    return aux.T
    
if __name__ == "__main__":
    
    X = pd.read_csv('x_input.csv', sep=",", header=None)
    Y = pd.read_csv('y_output.csv', sep=",", header=None)
    config = pd.read_csv('config.csv', sep=",", header=None)

    p = config.loc[0, 0] /100
    hn = config.loc[1, 0]
    mu = config.loc[2, 0]
    maxIter = config.loc[3, 0]

    a = 0.01
    b = 0.99
    
    D, N = X.shape
    L = round(N*p)
    
    X=random.shuffle(X)
    Y=random.shuffle(Y)
    
    x_normalizado = normalizar(X)
    y_normalizado = normalizar(Y)
    
    
    xe = x_normalizado.iloc[:, 0: L]
    ye = y_normalizado.iloc[:, 0: L]
    
    xv = x_normalizado.iloc[:, L:]
    yv = y_normalizado.iloc[:, L:]
    
    
 
    xe.to_csv(path_or_buf = 'train_x.csv', index = False , header = None)
    ye.to_csv(path_or_buf = 'train_y.csv', index = False , header = None)    
    
    xv.to_csv(path_or_buf = 'test_x.csv', index = False , header = None)
    yv.to_csv(path_or_buf = 'test_y.csv', index = False , header = None)