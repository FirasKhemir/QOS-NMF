import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.decomposition import NMF
from QOS_NMF import QOSNMF

Y,Z=[],[]
sizes = [i for i in range(10,500,5)]

for size in sizes:
    M = np.random.rand(size, size)
    M = np.abs(M)
    #sklearn NMF
    tps2_1 = time.time()
    model = NMF(n_components=size-1,max_iter=5000)
    model.fit(M)
    W = model.components_
    H = model.transform(M)
    tps2_2 = time.time()
    Y.append(tps2_2 - tps2_1)
    #QOS-NMF
    tps3_1 = time.time()
    W,H=QOSNMF(M,size)
    tps3_2 = time.time()
    Z.append(tps3_2 - tps3_1)

plt.plot(sizes,Y, label='sklearn NMF')
plt.plot(sizes,Z ,label='QOS-NMF')
plt.legend(loc='best')
plt.savefig("C:/Users/MSI/Desktop/comparaison.png")
plt.show()