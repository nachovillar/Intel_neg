import numpy as np
import pandas as pd
import math
import my_utility as ut

#Begining
def main():
    inp = "test_x.csv"
    out = "test_y.csv"
    file_w = 'pesos.npz'
    xv = ut.load_data_txt(inp)
    yv = ut.load_data_txt(out)
    w1, w2 = ut.load_w_npy(file_w)
    Act, zv = ut.ff_snn(xv, w1, w2)
    ut.metricasTest(Act[2], yv, zv)

if __name__ == '__main__':
    main()