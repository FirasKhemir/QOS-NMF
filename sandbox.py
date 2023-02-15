import numpy as np
from PIL import Image

#import image
image = Image.open('Lenna.png')
#make an array out of the image
M = np.asarray(image)
#make an image out of an array
data = Image.fromarray(M)
#save the image tp the file
#data.save('output.png')
print(M)