import numpy as np
import numpy.linalg as linalg
import sys

def COSNMF(M,r):
    Maxiter = 500
    U,S,V = linalg.svd(M)
    Vbarre = V[:,:r]
    Ro = float('inf')
    t = 0
    while 1:
        t += 1
        VbarrePlus = np.where(Vbarre < 0, 0, Vbarre)
        Y = np.dot( Vbarre.T, VbarrePlus)
        Ro0 = Ro
        Ro = linalg.norm(Y-np.identity(Y.shape[0]),'fro')**2
        if(Ro0-Ro < sys.float_info.epsilon) or (t == Maxiter):
            break
        U,S,V= linalg.svd(Y)
        Vbarre = np.dot(Vbarre,np.dot(U,V.T))  
    S = np.dot(Vbarre,Y)
    aux = np.dot(M,Vbarre)
    W = np.dot(aux,linalg.inv(Y.T)) 
    return W , S