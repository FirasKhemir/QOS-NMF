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

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    print("topic #%d" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:10 - 1:-1]]))
    print()
    
for i, doc in enumerate(corpus):
    print("Document #%d" % i)
    print(doc)
    print("Topic distribution:", nmf.transform(tfidf[i])[0])
    print()
    
model = NMF(n_components=topicNbr, init='random', random_state=0)
W = model.fit_transform(M)
H = model.components_ 
print("M:\n ",M)
print("W:\n ",W)
print("H:\n ",H.T)

W1,H1=COSNMF(M,topicNbr)
print("W1: \n",W1) 
print("H1: \n",H1)

 
for topic_idx, topic in enumerate(nmf.components_):
    print(topic_idx,topic)    
    
for i, doc in enumerate(corpus):
    print(i,doc)