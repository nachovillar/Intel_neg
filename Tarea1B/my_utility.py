import numpy as np
import pandas as pd

#Load parameters of de SNN
def load_config():
    par = np.genfromtxt("config.csv", delimiter=',')
    param = []
    param.append(np.int8(par[0]))
    param.append(np.int8(par[1]))
    param.append(np.float(par[2]))
    param.append(np.int_(par[3]))
    return(param)

def load_data_txt(filename):
    x = pd.read_csv(filename, header=None)
    x = np.array(x)
    return(x)

def iniW(hn, n0):
    w = np.random.rand(hn, n0)
    x = hn + n0
    r = np.sqrt(6/x)
    w = w*2*r - r
    return(w)

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
    np.savez("pesos.npz", w1=w1, w2=w2)
    np.savetxt("train_costo.csv", mse, delimiter='', fmt='%.6f')

def load_w_npy(file_w):
    arrays = np.load(file_w)

    w1 = arrays['w1']
    w2 = arrays['w2']
    return(w1, w2)

def ff_snn(x, w1, w2):
    a0 = x
    z = np.dot(w1, a0)
    a1 = 1 / (1 + np.exp(-z))
    zv = np.dot(w2, a1)
    a2 = 1 / (1 + np.exp(-zv))
    Act = [a0, a1, a2]
    return (Act)

def derivate_act(a):
    da = a*(1-a)
    return(da)

def fb_snn(a,ye,w1,w2,mu):
    e = a[2] - ye
    Cost = np.mean(e**2)
    dOut = e * derivate_act(a[2])
    gradW2 = np.dot(dOut, a[1].T)
    dHidden = (np.dot(w2.T, dOut)) * (derivate_act(a[1]))
    gradW1 = np.dot(dHidden, a[0].T)
    w2 = w2 - mu*gradW2
    w1 = w1 - mu*gradW1
    return(w1,w2,Cost)


def metricasTest(a2, yv):
    #ERROR DEL MODELO SNN
    err = yv - a2
    #MAE
    mae = (np.absolute(err)).mean()
    #MSE
    mse = (np.square(err)).mean()
    #RMSE
    rmse = np.sqrt(mse)
    #r2
    r2 = 1 - ((np.var(err)) / (np.var(a2)))
    
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2: ", r2*100)

    yz = []
    yz = np.hstack((yv.T, a2.T))
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