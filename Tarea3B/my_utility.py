# My Utility : auxiliars functions
import pandas as pd
import numpy  as np


#gets miniBatch
def get_miniBatch(i,x,bsize):
    z=x[:,i*bsize:(i+1)*bsize]
    return(z)

#STEP 1: Feed-forward of DAE
def forward_dl(x,w):
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

# STEP 2: Gradiente via BackPropagation
def grad_bp_dl(a,w):
    gradW = [None]*len(w)
    deltas = [None]*len(w)
    
    for idx in reversed(range(len(w)-1)):
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
    return(gradW)

# Update DL's Weight with Adam
def updW_Adam(w, p, q, gradiente, mu, iteracion):    
    b1 = 0.9
    b2 = 0.999
    e = 10**-8
    
    P = b1*p + (1-b1) * gradiente
    Q = b2*q + (1-b2) * gradiente**2
    gAdam = (np.square(1-b2**iteracion)/(1-b1**iteracion) ) * P/(np.square(Q + e))
    
    W = w - mu*gAdam
    return(W, P, Q)

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

# Softmax's gradient
def softmax_grad(x,y,w):
    z = np.dot(w,x)
    a = softmax(z)
    ya = y*np.log(a)
    #cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))
    return(gW)

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
def save_w_dl(W,npy_w,cost,csv_cost):
    np.savetxt(csv_cost, cost, delimiter=",")
    np.savez(npy_w, W=W)
    return()
    
#load weight of DL in numpy format
def load_w_dl(npy_w):
    w = np.load(npy_w, allow_pickle=True)
    return w['W']    
#
