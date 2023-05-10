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
print(A)
model = NMF(n_components=2, init='random', random_state=0)
W1 = model.fit_transform(P)
H1 = model.components_ 
# Step 3.2: Perform NMF using COS-NMF
W, H = QOSNMF(P, P.shape[0])
# Step 4: Extract the communities
n_communities = 2
kmeans = KMeans(n_clusters=n_communities, random_state=0, n_init=10).fit(W)
communities = kmeans.labels_
kmeans1 = KMeans(n_clusters=n_communities, random_state=0, n_init=10).fit(W1)
communities1 = kmeans1.labels_
print("Number of communities:", n_communities, "\nCommunities with QOS-NMF:", communities)
print("Number of communities:", n_communities, "\nCommunities with NMF:", communities1)

def extract_nodes(comm,n):
    nodes = []
    for i in range(len(comm)):
        if communities[i] == n:
            nodes.append(i+1)
    return nodes

node_lists = {}
for i in range(n_communities):
    node_lists[i] = extract_nodes(communities,i)
    
node_lists1 = {}
for i in range(n_communities):
    node_lists1[i] = extract_nodes(communities1,i)
    
pos = nx.spring_layout(G)

# graph without colors
nx.draw_networkx(G, pos, edge_color='k',  with_labels=True,font_weight='light', node_size=280, width=0.9)

# drawing each community with a different color
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
for i in range(n_communities):
    nx.draw_networkx_nodes(G, pos, nodelist=node_lists[i], node_color=colors[i])
plt.show()
for i in range(n_communities):
    nx.draw_networkx_nodes(G, pos, nodelist=node_lists1[i], node_color=colors[i])
plt.show()   

result = [i for i in communities]
result1 = [i for i in communities1]

print(result)
print(result1)
print(ground_truth)
