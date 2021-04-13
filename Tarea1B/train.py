import pandas as pd
import numpy as np
import math
import my_utility as ut



def train(xe, ye, param):

    w1, w2 = ut.iniW_snn(xe, ye, param[1], param[2])
    
    mse = []
    for iter in range(param[3]):
        Act = ut.ff_snn(xe, w1, w2)
        w1, w2, cost = ut.fb_snn(Act, ye ,w1, w2, param[2])
        mse.append(cost)
        if((iter % 200) == 0):
            print('iter: {:.5f}'.format(cost))
    return(w1, w2, mse)

# doing the complete rutine
def main ():
    inp = "train_x.csv"
    out = "train_y.csv"
    par_config = ut.load_config()
    xe = ut.load_data_txt(inp)
    ye = ut.load_data_txt(out)
    w1, w2, mse = train(xe, ye, par_config)
    ut.save_w_npy(w1, w2, mse)
    

if __name__ == '__main__':
    main()