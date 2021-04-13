import pandas as pd
import numpy as np
import math
import my_utility as ut



def train(xe, ye, param):
    
    hn = param[0]
    mu = param[1]
    maxIter = param[2]

    w1, w2 = ut.iniW_snn(xe, ye, hn, mu)
    mse = []
    for iter in range (maxIter):
        Act = ut.ff_snn(xe, w1, w2)
        w1, w2, cost = ut.fb_snn(Act, ye ,w1, w2, mu)
        mse.append(cost)
        if((iter % 200) == 0):
            print('iter:{:.5f}'.format(cost))
    return(w1, w2, mse)

# doing the complete rutine
def main ():
    inp = "train_x.csv"
    out = "train_y.csv"
    p, hn, mu, maxIter = ut.load_config()
    param = [hn, mu, maxIter]
    xe, ye = ut.load_data_txt(inp, out)
    w1, w2, mse = train(xe, ye, param)
    ut.save_w_npy(w1, w2, mse)

    a1, ze = ut.snn_ff(xe, w1, w2)
    ut.metricasTrain(a1, ye, ze)

if __name__ == '__main__':
    main()