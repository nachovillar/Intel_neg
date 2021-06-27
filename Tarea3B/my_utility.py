# My Utility : auxiliars functions
import pandas as pd
import numpy  as np


#gets miniBatch
def get_miniBatch(i,x,bsize):
    z=x[:,i*bsize:(i+1)*bsize]
    return(z)

#STEP 1: Feed-forward of DAE
def forward_dl(x,w):
	#completar code
    return(...)    

# STEP 2: Gradiente via BackPropagation
def grad_bp_dl(...):
    #completar code    
    return(...)    

# Update DL's Weight with Adam
def updW_Adam(...):    
    #completar code    
    return(...)

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))


# Init.weights of the DL 
def iniW():
    #256*512
    #128*256
    #256*128
    #512*256
    #10*512
    W = load_w_dl('w_dl.npz')
    P = []
    Q = []
    for n in W:
        P.append(np.zeros((n.shape[0],n.shape[1])))
        Q.append(np.zeros((n.shape[0],n.shape[1])))
    return(W,P,Q)

#Forward Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

# Métrica
def metricas(x,y):
    confussion_matrix = np.zeros((y.shape[0], x.shape[0]))
    
    for real, predicted in zip(y.T, x.T):
        confussion_matrix[np.argmax(real)][np.argmax(predicted)] += 1
        
    f_score = []
    
    for index, caracteristica in enumerate(confussion_matrix):
        
        TP = caracteristica[index]
        FP = confussion_matrix.sum(axis=0)[index] - TP
        FN = confussion_matrix.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append(2 * (precision * recall) / (precision + recall))
        
    metrics = pd.DataFrame(f_score)
    metrics.to_csv("metrica_dl.csv", index=False, header=False)
    return(f_score)
    

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the DL
def load_config():      
    par = np.genfromtxt("param_dl.csv",delimiter=',',dtype=None)    
    par_dl=[]    
    par_dl.append(np.float(par[0])) # Learn rate
    par_dl.append(np.int16(par[1])) # miniBatch Size
    par_dl.append(np.int16(par[2])) # MaxIter    
    return(par_dl)

# Load data 
def load_data_csv(fname):
    x     = pd.read_csv(fname, header = None)
    x     = np.array(x)  
    return(x)

#save weights of DL in numpy format
def save_w_dl(...):    
    #completar code    
    
#load weight of DL in numpy format
def load_w_dl(...):
    #completar code
    return(...)      
#
