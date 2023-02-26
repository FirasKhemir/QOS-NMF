from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('Lenna.png')
gray_img = image.convert("L")

M = np.asarray(image)
print(M.shape,M.size,M[0,0],M[0,0])

Mgray = np.asarray(gray_img)
print(Mgray.shape,Mgray.size,Mgray[0,0],Mgray[0,0].shape)

plt.imshow(gray_img, cmap='gray')
plt.show()