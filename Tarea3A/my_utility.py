# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))


#STEP 1: Feed-forward of DAE
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

# STEP 2: Feed-Backward
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
            
    return(gradW)                               #cost?

# Update DAE's weight with RMSprop
def updW_dae(w,v,gW,mu):    
    # completar code
        
    return(w,v)
#    
# Update Softmax's weight with RMSprop
def updW_softmax(w,v,gW,mu):    
    # completar code
    
    return(w,v)

# Initialize weights of the Deep-AE
#def ini_WV(...):
    # completar code
 #   return(W)

def ini_WV(input,nodesEnc):
    #print(input)
    #print(nodesEnc)
    W = []
    V = []
    prev = input
    #print("ITER1:")
    for n in range(len(nodesEnc)):
        W.append(randW(nodesEnc[n],prev))
        prev = nodesEnc[n]
    #print("ITER2:")
    for n in reversed(W):
        W.append(randW(n.shape[1],n.shape[0]))
    
    #print("outIter")
    
    V = np.copy(W)
    #print (type(W)) #tipo de dato lista?
    #Shape = np.shape(W)
    #print(Shape)
    
    '''
    print(W[0])
    print(np.shape(V[0]))
    V[0].fill(0)
    print(V[0])
    print(np.shape(V[0]))
    '''
    
    for n in V:
        n.fill(0)
    
    '''for n in V:
        print(n)
        print(np.shape(n))
    '''
    print(V)
    return(W,V)

# Initialize random weights
def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    #print("randW:")
    #print(w)
    #print(np.shape(w))
    return(w)

#Forward Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0,keepdims=True))

# Softmax's gradient
def softmax_grad(x,y,w):
    z = np.dot(w,x)
    a = softmax(z)
    ya = y*np.log(a)
    cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))
    return(gW,cost)
  
# Encoder
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
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt("param_dae.csv",delimiter=',',dtype=None)    
    par_sae=[]    
    par_sae.append(np.float(par[0])) # Learn rate
    par_sae.append(np.int16(par[1])) # miniBatchSize
    par_sae.append(np.int16(par[2])) # MaxIter
    for i in range(3,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt("param_softmax.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning     
    return(par_sae,par_sft)

# Load data 
def load_data_csv(fname):
    x     = pd.read_csv(fname, header = None)
    x     = np.array(x)  
    return(x)

# save weights of the DL in numpy format
#W,Ws,'w_dl.npz',cost,'costo_softmax.csv'
#def save_w_dl(W,Ws,nfile_w,cost,nfile_sft):    
    # cmpletar code
    
def save_w_dl(W,Ws,npy_w,cost,csv_cost):
    np.savetxt(csv_cost, cost, delimiter=",")
    W.append(Ws)
    np.savez(npy_w, W=W)
    return()
    
 
#load weight of the DL in numpy format
def load_w_dl(npy_w):
    w = np.load(npy_w, allow_pickle=True)
    return w['W']

# save weights in numpy format
def save_w_npy(w1,w2,mse):  
    # completar code
    return()
