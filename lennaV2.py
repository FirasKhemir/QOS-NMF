import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from QOS_NMF import QOSNMF

#r = 512 #512 (256 )

image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img) 

""" 
for r in [512,510,500,475,450,400,350,300,250,100]:
    W,S=COSNMFV2(M,r)
    out = np.dot(W,S.T)

    data = Image.fromarray(out)
    plt.imshow(data, cmap='gray')
    #plt.show()
    new_p = data.convert("L")
    new_p.save("LennaOutput"+str(r)+"NMFV2.png")
    #loss = mean_absolute_error(out, M)    #mean square error
    #print(loss)

 """
 
W,S=QOSNMF(M,450)
out = np.dot(W,S.T)

data = Image.fromarray(out)
w_img = Image.fromarray(W)
s_img = Image.fromarray(S)
#plt.imshow(data, cmap='gray')
#plt.show()
plt.imshow(w_img, cmap='gray')
plt.show()
#plt.imshow(s_img, cmap='gray')
#plt.show()

    #loss = mean_absolute_error(out, M)    #mean square error
    #print(loss)