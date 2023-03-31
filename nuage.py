import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from cos_nmf import COSNMF

# Générer des données de nuage de points aléatoires
X = np.random.rand(100, 2)

# Appliquer COSNMF
W, H = COSNMF(X, X.shape[0])

# Assigner des couleurs à chaque cluster
colors = ['r', 'g', 'b']
labels = np.argmax(W, axis=1)
colors = [colors[label] for label in labels]

# Afficher le nuage de points avec des couleurs pour chaque cluster
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()