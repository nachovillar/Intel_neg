import numpy as np
import pandas as pd
import math
import pickle  #importamos uno mas



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
    #escribir el resutado
    #guardar w1,w2 en un csv se pueden juntar las 2 matrices para facilitar la operacion
    #para machine learning hoy en dia se ocupan los pickles para guardar arrays
    pesos = open("pesos.pickle", "wb")
    pickle.dump(w1, pesos)
    pickle.dump(w2, pesos)
    pesos.close()
    
    #pd.read_csv(inp, sep=",", header=None)
    

def load_w_npy(filename):
    #pesos_load = pickle.load(open("pesos.pickle", "rb"))
    pesos = open("pesos.pickle")
    w1= pickle.load("pesos.pickle")
    w2= pickle.load("pesos.pickle")
    pesos.close
    return(w1, w2)
