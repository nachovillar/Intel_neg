# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

# Calculate Pseudo-inverse
def pinv_ae(x,w1,c): 
    H = act_sigmoid(np.dot(w1,x))
    I=np.identity(H.shape[0])
    a=np.dot(x,H.T)
    b=np.dot(H,H.T)
    d=b+I/c
    pinv=np.linalg.pinv(d)
    w2 = np.dot(a,pinv)
    return w2

#AE's Feed-Backward
def backward_ae(x,w1,w2,mu):
    a = forward_ae(x,w1,w2)
    E = x-a[2]
    Delta_out = E
    Delta_hidden = np.multiply(np.dot(w2.T, Delta_out), derivate_act(act_sigmoid(a[1])))
    gradW1=np.dot(Delta_hidden,a[0].T)
    w1=w1+mu*gradW1
    return(w1)     

def forward_ae(x,w1,w2):
    w=[w1,w2]
    A=[x]
    XWN=np.dot(w[0],A[0])
    a=act_sigmoid(XWN)
    A.append(a)
    a=np.dot(w2,a)
    A.append(a)
    return A

    
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def derivate_act(a):
    return(a*(1-a))


#Forward Softmax
def softmax(x,w):
    z=np.dot(w,x)
    e = np.exp(z-np.max(z, axis=0, keepdims=True))
    y=e/e.sum(axis=0, keepdims=True)
    return(y)

# Softmax's gradient
def softmax_grad(x,T,w,lamb):
    N     = x.shape[1]
    y     = softmax(x,w)
    ty    = np.multiply(T,np.log10(y))  # Cross Entropy
    Costo     = (-1/N)*np.sum(np.sum(ty))
    Costo     = Costo+((lamb/2)*((np.linalg.norm(w, ord=2))**2))
    gradW = ((-1.0/N)*(np.dot((T-y),np.transpose(x))))+(lamb*w)
    return gradW,Costo


# Initialize weights
def iniW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)


#Measure
#def measure( ):        
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
def save_w_dl(W,Ws,cost, nombre_pesos_dl, nombre_costos_softmax_csv):    
    np.savez(nombre_pesos_dl, W=W, Ws=Ws)
    np.savetxt(nombre_costos_softmax_csv, cost, delimiter='', fmt='%.6f')
    
    

#load weight of the DL in numpy 
def load_w_dl(file_w):
    arrays = np.load(file_w)
    W = arrays['W']
    return(W)    
