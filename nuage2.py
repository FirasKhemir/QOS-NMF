import numpy as np
import matplotlib.pyplot as plt
from cos_nmf import COSNMF

""" 
X = np.random.rand(100, 2)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
pos_X = np.where(X >= 0, X, -X)

W, H = COSNMF(X, X.shape[0]) 
"""

# Créer les données
x = np.random.randn(100)
y = np.random.randn(100)
mask = (x > 0) & (y > 0)
x = x[mask]
y = y[mask]

# Tracer le nuage de points
plt.scatter(x, y)
plt.show()

z = np.column_stack((x, y))
print(z)
W, H = COSNMF(z, z.shape[0])

colors = ['r', 'g', 'b']
labels = np.argmax(W, axis=1)
colors = [colors[label] for label in labels]

# Afficher le nuage de points avec des couleurs pour chaque cluster
plt.scatter(z[:, 0], z[:, 1], c=colors)
plt.show()
