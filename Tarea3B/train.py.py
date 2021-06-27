# Fase 2: Deep-Learning:Training via Adam

import pandas     as pd
import numpy      as np
import my_utility as ut
	
#Training: Deep Learning
def train_dl(x,y,param):
    W,P,Q     = ut.iniW()    
    numBatch = np.int16(np.floor(x.shape[1]/param[1]))    
    cost = []
    #completar code
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

