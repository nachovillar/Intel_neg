# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

# Calculate Pseudo-inverse
def pinv_ae(): 
    #Completar code    
    return()

#AE's Feed-Backward
def backward_ae():
    #Completar code    
    return()    

    
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def derivate_act(a):
    return(a*(1-a))


#Forward Softmax
def softmax():
    #Completar code...        
        return()


# Softmax's gradient
def softmax_grad():
    #Completar  code...
    return()


# Initialize weights
def iniW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)


#Measure
def measure( ):        
        #Completar code..        
        
#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the DL: 
def load_config():      
    par = np.genfromtxt("param_sae.csv",delimiter=',')    
    par_sae=[]
    par_sae.append(np.float(par[0])) # % train
    par_sae.append(np.float(par[1])) # Learn rate
    par_sae.append(np.int16(par[2])) # Penal. C
    par_sae.append(np.int16(par[3])) # MaxIter
    for i in range(4,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt("param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning rate
    par_sft.append(np.float(par[2]))   #Lambda
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)

# save weights of the DL in numpy 
def save_w_dl():    
    #Completar code...
    
    

#load weight of the DL in numpy 
def load_w_dl():
        #Completar code. ..
    return()    
