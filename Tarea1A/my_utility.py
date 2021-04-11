import numpy as np
import pandas as pd

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

def save_w_npy(w1, w2):
    np.savez("pesos", w1, w2)

def load_w_npy(file_w):
    arrays = np.load(file_w)
    w1 = arrays['arr_0']
    w2 = arrays['arr_1']
    return w1,w2

def snn_ff(xv,w1,w2):
    z = np.dot(w1, xv)
    a1 = 1/(1+np.exp(-z))
    zv = np.dot(w2, a1)
    a2 = 1/(1+np.exp(-zv))
    
    return a2, zv

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