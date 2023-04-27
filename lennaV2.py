import numpy as np
import numpy.linalg as linalg
import sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from cos_nmf_V2 import COSNMFV2

r = 512 #512 (256 )

image = Image.open('Lenna.png')
gray_img = image.convert("L")
M = np.asarray(gray_img) 

W,S=COSNMFV2(M,r)
out = np.dot(W,S.T)

data = Image.fromarray(out)
plt.imshow(data, cmap='gray')
plt.show()
new_p = data.convert("L")
new_p.save("output"+str(r)+"NMFV2.png")
loss = mean_absolute_error(out, M)    #mean square error
print(loss)