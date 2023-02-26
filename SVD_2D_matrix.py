from numpy import array
from scipy.linalg import svd
import numpy as np

A = array([[1, 2], [3, 4], [5, 6]]) #starting matrix
print("matrice A\n",A)
# SVD
U, s, VT = svd(A)
print("matrice U\n",U)
print("vecteur s\n",s)
print("matrice VT\n",VT)

smat = np.zeros(A.shape) #creating a matrix with same dimensions as A

#smat[:2, :2] = np.diag(s) #the value 2 is entered manually
#or also we can do this
smat[:s.shape[0], :s.shape[0]] = np.diag(s) 
# which takes values from shape
print("matrice Produit U.s.VT: \n",np.dot(U, np.dot(smat, VT)))
