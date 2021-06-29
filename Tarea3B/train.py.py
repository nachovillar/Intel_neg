# Fase 2: Deep-Learning:Training via Adam

import pandas     as pd
import numpy      as np
import my_utility as ut

def train_dae(x, W, P, Q,mu,numBatch,BatchSize):
    for i in range(numBatch):
        xe   = ut.get_miniBatch(i,x,BatchSize)
        Act  = ut.forward_dl(xe,W)
        gW   = ut.grad_bp_dl(Act,W)
        W, P, Q  = ut.updW_Adam(W, P, Q, gW, mu, i);  #eReMeSPPROP
    return(W)



#Training: Deep Learning
def train_dl(x,y,param):
    W,P,Q     = ut.iniW()    
    numBatch = np.int16(np.floor(x.shape[1]/param[1]))    
    cost = []
    for i in range(param[2]):
        xe  = x[:,np.random.permutation(x.shape[1])] 
        W   = train_dae(xe, W,P,Q,param[0],numBatch,param[1])
 
    return(W, cost) 
   
# Beginning ...
def main():
    par_dl          = ut.load_config()    #param_dl
                                            #LearnRate 0.01
                                            #miniBatchSize 32
                                            #MaxIter 100
    xe              = ut.load_data_csv('train_x.csv')    
    ye              = ut.load_data_csv('train_y.csv')    
    W, cost         = train_dl(xe,ye,par_dl)         
    ut.save_w_dl(W,'w_dl2.npz',cost,'costo_dl.csv')
       
if __name__ == '__main__':   
	 main()
