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
    # HACER FUNCION QUE GUARDA AMBAS MATRICES EN UN ARCHIVO
    return

def load_w_npy(filename):
    
    print(w1, w2)