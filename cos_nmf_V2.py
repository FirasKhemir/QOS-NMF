from scipy.optimize import nnls
import sys
import numpy.linalg as linalg
import numpy as np

def COSNMFV2(M,r):
    eps=sys.float_info.epsilon
    maxIter= 1
    t = 0
    U,D,V = linalg.svd(M)

    Vr = V[:,:r]
    I=np.identity(r)

    while 1:
        t=t+1
        for j in range (0,r):
            L, x= nnls(-Vr.T, I[:,j])
        

        Y = I + np.dot(Vr.T,L)
        linalg.det(Y)
        rho= linalg.norm(Y-I,'fro')

        if (rho<eps) or (t>maxIter):
            [t,rho]
            break

        U,D,V=linalg.svd(Y)
        Vr= np.dot(Vr,np.dot(U,V.T))

    S = np.dot(Vr,Y)
    W=np.dot(M,np.dot(Vr,linalg.inv(Y.T)))
    return W , S