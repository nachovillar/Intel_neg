import numpy as np
import pandas as pd

#Load parameters of de SNN
def load_config():
    aux = pd.read_csv('config.csv', sep=",", header=None)
    p = aux.loc[0, 0] /100  #% de training
    hn = aux.loc[1, 0]      #Hiden notes
    mu = aux.loc[2, 0]
    maxIter = aux.loc[3, 0]   #Maximo de Iteraciones 

    return (p,hn,mu,maxIter)

def load_data_txt(inp, out):
    xePd = pd.read_csv(inp, sep=",", header=None)
    yePd = pd.read_csv(out, sep=",", header=None)
    xe = xePd.to_numpy(dtype='float', na_value=np.nan)
    ye = yePd.to_numpy(dtype='float', na_value=np.nan)
    return (xe, ye)

def iniW(hn, n0):
    w = np.random.random((hn, n0))
    x = hn + n0
    r = np.sqrt(6/x)
    w = w*2*r - r
    return w

#Pseudo inverse
def p_inversa(a1, ye, hn, mu):
    yh = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(hn)/mu
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(yh, p_inv)
    return(w2)

def iniW_snn(xe, ye, hn, mu):
    n0 = xe.shape[0]      #numero de nodos de entrada
    w1 = iniW(hn, n0)    
    z = np.dot(w1, xe)
    a1 = 1/(1 + np.exp(-z))
    w2 = p_inversa(a1, ye, hn, mu)
    return(w1, w2)

def save_w_npy(w1, w2, mse):
    np.savez("pesos", w1, w2, mse)

def load_w_npy(file_w):
    arrays = np.load(file_w)
    w1 = arrays['arr_0']
    w2 = arrays['arr_1']
    return w1,w2

def ff_snn(x, w1, w2):
    a0 = x
    z = np.dot(w1, x)
    a1 = 1/(1+np.exp(-z))
    zv = np.dot(w2, a1)
    a2 = 1/(1+np.exp(-zv))
    Act = [a0, a1, a2]
    return Act, zv

def metricasTest(a2, yv, zv):
    #ERROR DEL MODELO SNN
    err = yv - a2
    #MAE
    mae = (np.absolute(err)).mean()
    #MSE
    mse = (np.square(err)).mean()
    #RMSE
    rmse = np.sqrt(mse)
    #r2
    r2 = 1 - ((np.var(a2)) / (np.var(yv)))
    
    print("MAE: ",mae)
    print("MSE: ",mse)
    print("RMSE: ",rmse)
    print("R2: ",r2)

    yz = []
    yz = np.hstack((yv.T, zv.T))
    np.savetxt("test_costo.csv", yz, delimiter=' ',fmt='%.6f')
    

    vecto = np.array([ mae, mse, rmse, r2 ])
    pd.DataFrame(vecto).to_csv("test_metrica.csv", header=None, index=None)
    
def metricasTrain(a1, ye, ze):
    #ERROR DEL MODELO SNN
    err = ye - a1
    #MAE
    mae = (np.absolute(err)).mean()
    #MSE
    mse = (np.square(err)).mean()
    #RMSE
    rmse = np.sqrt(mse)
    #r2
    r2 = 1 - ((np.var(a1)) / (np.var(ye)))
    
    print("MAE: ",mae)
    print("MSE: ",mse)
    print("RMSE: ",rmse)
    print("R2: ",r2)
 
    yz = []
    yz = np.hstack((ye.T, ze.T))
    np.savetxt("train_costo.csv", yz, delimiter=' ',fmt='%.6f')
    
    vecto = np.array([ mae, mse, rmse, r2 ])
    pd.DataFrame(vecto).to_csv("train_metrica.csv", header=None, index=None)