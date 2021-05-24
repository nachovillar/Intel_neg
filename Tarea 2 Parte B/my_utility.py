# My Utility : auxiliars functions
import pandas as pd
import numpy  as np


def forward_dae(x,w):	
    Act=[]
    a0=x
    
    z=np.dot(w[0],x)
    a1=act_sigmoid(z)
    
    Act.append(a0)
    Act.append(a1)
    
    ai=a1
    
    for i in range(len(w)):
        if i != 0:
            zi=np.dot(w[i],ai)
            ai=act_sigmoid(zi)
            Act.append(ai)
    return(Act)    

def grad_bp_dae(a,w):
    gradW = [None]*len(w)
    deltas = [None]*len(w)
    
    for idx in reversed(range(len(w))):
        if(idx != (len(w)-1)):
            delta_next = deltas[idx+1]
            
            delta_ = np.dot(w[idx+1].T, delta_next)
            da = deriva_sigmoid(a[idx+1])
            
            deltaH = delta_ * da
            
            grad = np.dot(deltaH,a[idx].T)
            
            gradW[idx] = grad
            deltas[idx] = deltaH
        else:
            e= a[-1]-a[0]
            da = deriva_sigmoid(a[-1])
            
            delta_f = e*da
            
            grad = np.dot(delta_f,a[-2].T)
            
            gradW[-1] = grad
            deltas[-1] = delta_f
            
    return(gradW) #sacaron el retorno de cost     

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

def updW_sgd(w,gradW,mu):
    for i in range(len(w)):
        tau = mu/len(w)
        mu_k = mu/(1+np.dot(tau,(i+1)))
        w[i] = w[i] - mu_k*gradW[i]
    return(w)
#    
def iniW(input,nodesEnc):
    W = []
    prev = input
    for n in range(len(nodesEnc)):
        W.append(randW(nodesEnc[n],prev))
        prev = nodesEnc[n]
    for n in reversed(W):
        W.append(randW(n.shape[1],n.shape[0]))
    return(W)

# Initialize random weights
def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)

#Forward Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z/exp_z.sum(axis=0, keepdims=True)

# Softmax's gradient
def softmax_grad(x,y,w,lambW):
    z = np.dot(w,x)
    a = softmax(z)
    ya = y*np.log(a)
    cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    cost = cost + lambW/(2*np.linalg.norm(w,2))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))+lambW
    return(gW,cost)
  
def encoder(x,w):
    for weight in w:
        x = act_sigmoid(np.dot(weight,x))
    return(x)

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
    return(par_dae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)


def save_w_dl(W,Ws,cost,npy_w,csv_cost):
    np.savetxt(csv_cost, cost, delimiter=",")
    W.append(Ws)
    np.savez(npy_w, W=W)

    return()

def load_w_dl(npy_w):
    w = np.load(npy_w, allow_pickle=True)
    return w['W']
    
