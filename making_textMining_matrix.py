from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF 
from cos_nmf import COSNMF
import numpy as np
import sys 

corpus = ["this is the first document it is talking about the industrial revolution",
          "this is the second document in which we discuss the importance of oral hygene",
          "within this third document we will talk about the effect of radio active waves on the human body",
          "lastly we would like to discuss the drawbacks of smoking of course"]

topicNbr = 2

vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(corpus)
M=tfidf.toarray()
nmf = NMF(n_components=topicNbr,random_state=1)
nmf.fit(tfidf)

np.set_printoptions(threshold=sys.maxsize)
print(M)