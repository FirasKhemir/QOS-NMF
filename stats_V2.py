import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from cos_nmf_V2 import COSNMFV2


image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img)

A,B=[],[]
for i in range(512,300,-10):
    r = i
    W,S=COSNMFV2(M,r)
    out = np.dot(W,S.T)
    loss = mean_absolute_error(out, M)    #mean square error
    A.append(i)
    B.append(loss)
plt.plot(A,B)
plt.gca().invert_xaxis()
plt.savefig('loss_curve.png')
plt.show()