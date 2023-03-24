import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt
r = 480 #512 (256 )
Maxiter = 500
Ro = float('inf')
t = 0
image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img) 
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
#S = np.where(Vbarre < 0, 0, Vbarre) #Vbarre Plus
S = np.dot(Vbarre,Y)
aux = np.dot(M,Vbarre)
W = np.dot(aux,linalg.inv(Y.T)) # transposé de la matrice diagonale inferieure de Y
                                # Il s'agit de l'inverse de la transposée de la matrice Y
out = np.dot(W,S.T)

data = Image.fromarray(out)
plt.imshow(data, cmap='gray')
plt.show()
new_p = data.convert("L")
new_p.save("output"+str(r)+".png")