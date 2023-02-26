import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt

r = 512
Maxiter = 500

Ro = float('inf')
t = 0

image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img) 

print("M shape", M.shape)
U,S,V = linalg.svd(M)
print("V shape", V.shape)
Vbarre = V[:,:r]

Ro = float('inf')
t = 0
while 1:
    t += 1
    Y = np.dot( Vbarre.T, Vbarre)
    Ro0 = Ro
    Ro = linalg.norm(Y-np.identity(Y.shape[0]),'fro')**2
    if(Ro0-Ro < sys.float_info.epsilon) or (t == Maxiter):
        break
    U,S,V= linalg.svd(Y)
    Vbarre = np.dot(Vbarre,np.dot(U,V.T))
    
S = Vbarre
print("S shape", S.shape)
print(Vbarre.shape)
print(M.shape)
print(np.tril(Y).T.shape)
aux = np.dot(M,Vbarre)
W = np.dot(aux,linalg.inv(Y.T)) # transposé de la matrice diagonale inferieure de Y
                                # Il s'agit de l'inverse de la transposée de la matrice Y

out = np.dot(W,S.T)
data = Image.fromarray(out)

plt.imshow(data, cmap='gray')
plt.show()
#data.save('please.png')