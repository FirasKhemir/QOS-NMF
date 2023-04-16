import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from cos_nmf import COSNMF


# Générer des données de nuage de points aléatoires
X = np.random.rand(100, 2)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Appliquer COSNMF
#W, H = COSNMF(X, X.shape[0])

# Appliquer NMF de Sklearn
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

# Assigner des couleurs à chaque cluster
colors = ['r', 'g', 'b']
labels = np.argmax(W, axis=1)
colors = [colors[label] for label in labels]

# Afficher le nuage de points avec des couleurs pour chaque cluster
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()
 

# Créer les données
x = np.random.randn(100)
y = np.random.randn(100)
mask = (x > 0) & (y > 0)
x = x[mask]
y = y[mask]

# Tracer le nuage de points
plt.scatter(x, y)
#plt.title('Nuage de points')
plt.xlabel('Axe X')
plt.ylabel('Axe Y')

z = np.column_stack((x, y))
model = NMF(n_components=2, init='random', random_state=0)
W1 = model.fit_transform(z)
H1 = model.components_

colors = ['r', 'g', 'b']
labels = np.argmax(W1, axis=1)
colors = [colors[label] for label in labels]
plt.scatter(z[:, 0], z[:, 1], c=colors)
plt.show()