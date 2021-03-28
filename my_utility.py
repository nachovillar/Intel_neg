import numpy as np
import pandas as pd
import math

def load_data_txt(inp, out):
    xe = pd.read_csv(inp, sep=",", header=None)
    ye = pd.read_csv(out, sep=",", header=None)
    return (xe, ye)

def iniW(hn, n0):
    w = np.random.random((hn, n0))
    x = hn + n0
    r = np.sqrt(6/x)
    w = w*2*r - r
    return w

def save_w_npy(w1, w2):
    np.savez("pesos", w1=w1, w2=w2)
    
def load_w_npy(filename):
    container = np.load(filename)
    weight_data = [container[key] for key in container]
    w1 = weight_data[0]
    w2 = weight_data[1]
    return(w1, w2)

def snn_ff(xv, w1, w2):
    z = np.dot(w1, xv)
    a1 = 1/(1 + np.exp(-z))
    zv = np.dot(w2, a1)
    return zv
    

def metricas(yv, zv):