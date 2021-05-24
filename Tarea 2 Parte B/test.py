import pandas as pd
import numpy as np
import my_utility as ut


def forward_dl(xv,W): 
    L=len(W)
    x=xv
    for i in range(L-1):
        x = ut.act_sigmoid(np.dot(W[i],x))
    zv = ut.softmax(np.dot(W[L-1],x))
    return(zv)

# Beginning ...
def main():		
	xv     = ut.load_data_csv('test_x.csv')	
	yv     = ut.load_data_csv('test_y.csv')	
	W      = ut.load_w_dl('w_dl.npz')
	zv     = forward_dl(xv,W)      		
	Fsc    = ut.metricas(yv,zv) 	

if __name__ == '__main__':   
	 main()

