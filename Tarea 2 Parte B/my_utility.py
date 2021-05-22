# My Utility : auxiliars functions
import pandas as pd
import numpy  as np


def forward_dae(x,w):	
    #completar....
    return(a)    


def grad_bp_dae(a,w):
    #completar...
    return(gradW, Cost)    

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

def updW_sgd(w,gradW,mu):
    #completar....
    return(w)
#    
def iniW(input,nodesEnc):
    #completar...
    return(W)

# Initialize random weights
def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)

#Forward Softmax
def softmax(z):
        #completar
        return()

# Softmax's gradient
def softmax_grad(x,y,w,lambW):
    #completar...  
    return(gW,Cost)
  
def encoder(x,w):
    #completar..    
    return(act)
# Métrica
def metricas(x,y):
    #completar....
    return(Fscore)
   

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
def load_config():      
    par = np.genfromtxt("param_dae.csv",delimiter=',')    
    par_dae=[]    
    par_dae.append(np.float(par[0])) # Learn rate
    par_dae.append(np.int16(par[1])) # miniBatchSize
    par_dae.append(np.int16(par[2])) # MaxIter
    for i in range(3,len(par)):
        par_dae.append(np.int16(par[i]))
    par    = np.genfromtxt("param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning 
    par_sft.append(np.float(par[2]))   #Lambda
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)


def save_w_dl(W,Ws,nfile_w,cost,nfile_sft):    
#completar...    

def load_w_dl(nfile):
    #completar...
    return(W)    
    
