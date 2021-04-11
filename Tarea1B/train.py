import pandas as pd
import numpy as np
import math
import my_utility as ut

#Pseudo inverse
def p_inversa(a1, ye, hn, C):
    yh = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(hn)/C
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(yh, p_inv)
    return(w2)

#Training SNN via Pseudo_inverse
def train_snn(xe, ye, hn, mu, maxIter):
    w1, w2 = ut.iniW_snn(xe.shape[0], param[1], 1)
    for iter in range (param[3]):
        Act = ut.ff_snn(xe, w1, w2)
        w1, w2, cost = ut.fb_snn(Act, ye ,w1, w2, param[2])
        mse.append(cost)
        if((iter % 200) == 0):
            print('iter:{:.5f}'.format(cost))
    return(w1, w2, mse)

    #n0 = xe.shape[0]        #numero de nodos de entrada
    #w1 = ut.iniW(hn, n0)    
    #z = np.dot(w1, xe)
    #a1 = 1/(1 + np.exp(-z))
    #w2 = p_inversa(a1, ye, hn, C)
    
    return(w1, w2)

#Load parameters of de SNN
def load_config():
    aux = pd.read_csv('config.csv', sep=",", header=None)
    p = aux.loc[0, 0] /100  #% de training
    hn = aux.loc[1, 0]      #Hiden notes
    mu = config.loc[2, 0]
    maxIter = config.loc[3, 0]   #Maximo de Iteraciones 

    return (p,hn,mu,maxIter)

#Starter packkk
def main ():
    inp = "train_x.csv"
    out = "train_y.csv"
    p,hn,mu,maxIter = load_config()
    xe, ye = ut.load_data_txt(inp, out)
    w1, w2 = train_snn(xe, ye, hn, C)
    ut.save_w_npy(w1,w2)

    a1, ze = ut.snn_ff(xe, w1, w2)
    ut.metricasTrain(a1, ye, ze)

if __name__ == '__main__':
    main()