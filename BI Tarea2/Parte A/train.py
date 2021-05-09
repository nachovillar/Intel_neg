# Deep-Learning:Training via BP+Pseudo-inverse

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    #Completar code...
    return(w,costo)

# AE's Training 

def train_ae(x,hnode,param):
    #completar code ...
    return(w1)

def train_sae(x,param):
    W={}
    for hn in range(4,len(param)):   #Number of AEs     
        w1       = train_ae(x,hn,param)
        W[hn-4]  = w1
        x        = ut.act_sigmoid(np.dot(w1,x))         
    return(W,x) 
   
# Beginning ...
def main():
    par_sae,par_sft = ut.load_config()    
    xe              = ut.load_data_csv('train_x.csv')
    ye              = ut.load_data_csv('train_y.csv')
    W,Xr            = train_sae(xe,par_sae) 
    Ws, cost        = train_softmax(Xr,ye,par_sft)
    ut.save_w_dl(W,Ws,cost,'w_dl.npz','cost_sofmax.csv')
       
if __name__ == '__main__':   
	 main()

