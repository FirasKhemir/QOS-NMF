import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image

r = 4
Maxiter = 500

Ro = float('inf')
t = 0

#image = Image.open('Lenna.png')
#M = np.asarray(image)

M= np.random.randint(100, size=(5,5))
print(M)
print("M shape", M.shape)
U,S,V = linalg.svd(M)
print(V)
print("V shape", V.shape)
Vbarre = V[:,:r]

while 1:
    t += 1
    Y = np.dot( Vbarre.T, np.triu(Vbarre) )
    Ro0 = Ro
    Ro = linalg.norm(Y-np.identity(Y.shape[0]),'fro')**2
    if(Ro0-Ro < sys.float_info.epsilon) or (t == Maxiter):
        break
    U,S,V= linalg.svd(Y)
    Vbarre = np.dot(Vbarre,U.dot(V.T))
    
S = np.triu(Vbarre)
print("S shape", S.shape)
print(Vbarre.shape)
print(M.shape)
print(np.tril(Y).T.shape)
aux = np.dot(M,Vbarre)
W = np.dot(aux,linalg.inv(Y.T)) # transposé de la matrice diagonale inferieure de Y


out = np.dot(W,S.T)
print("M",M)
print("out",out)

""" out = np.multiply(W,S)
data = Image.fromarray(out)

data.save('please.png') """