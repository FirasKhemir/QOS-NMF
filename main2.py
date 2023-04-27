import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import nnls
#from scipy.optimize import lsqnonneg
import time
from lsqnonnegMatlabVersion import lsqnonneg

r = 512
eps=sys.float_info.epsilon
maxIter= 1
t = 0
I=np.identity(r)
image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img)
U,D,V = linalg.svd(M)
Vr = V[:,:r]

prev = time.time()
while 1:
    print("line 1 took ",time.time()-prev," seconds")
    prev=time.time()
    t=t+1
    print("line 2 took ",time.time()-prev," seconds")
    prev=time.time()
    for j in range (0,r):
        #L, x= nnls(-Vr.T, I[:,j]) #built in version
        #L, x=lsqnonneg(-Vr.T, I[:,j]) #scipy version
        L, x,residual=lsqnonneg(-Vr.T, I[:,j])
        
    print("line 3 took (j boucle) ",time.time()-prev," seconds")
    prev=time.time()
    Y = I + np.dot(Vr.T,L)
    print("line 4 took (y = sum of i and dot)",time.time()-prev," seconds")
    prev=time.time()
    linalg.det(Y)
    print("line 5 took det calculation",time.time()-prev," seconds")
    prev=time.time()
    rho= linalg.norm(Y-I,'fro')
    print("line 6 took (linalg norm)",time.time()-prev," seconds")
    prev=time.time()
    if (rho<eps) or (t>maxIter):
        [t,rho]
        break
    print("line 7 took (if condition)",time.time()-prev," seconds")
    prev=time.time()
    U,D,V=linalg.svd(Y)
    print("line 8 took ",time.time()-prev," seconds")
    prev=time.time()
    Vr= np.dot(Vr,np.dot(U,V.T))
print("line 9 took (main while 1)",time.time()-prev," seconds")
prev=time.time()
S = np.dot(Vr,Y)
W=np.dot(M,np.dot(Vr,linalg.inv(Y.T)))
out = np.dot(W,S.T)
data = Image.fromarray(out)
plt.imshow(data, cmap='gray')
plt.show()
print("total runtime ",time.time()-prev," seconds")