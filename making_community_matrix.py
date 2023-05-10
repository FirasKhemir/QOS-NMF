from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF 
from cos_nmf import COSNMF
import sys
import networkx as nx
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from QOS_NMF import QOSNMF
import matplotlib.pyplot as plt


ground_truth = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
G = nx.read_gml("karate.gml", label='id')
# seeing the graph
nx.draw_networkx(G, with_labels=True)
plt.show()
# Construct the adjacency matrix
A = nx.to_numpy_array(G)
# Step 2: Normalize the adjacency matrix
P = A / np.sum(A, axis=1, keepdims=True)
# Step 3.1: Perform NMF using Sklearn
np.set_printoptions(threshold=sys.maxsize)
print(A)