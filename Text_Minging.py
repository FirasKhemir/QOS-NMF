from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF 
from cos_nmf import COSNMF

corpus = ["this is the first document it is talking about the industrial revolution",
          "this is the second document in which we discuss the importance of oral hygene",
          "within this third document we will talk about the effect of radio active waves on the human body",
          "lastly we would like to discuss the drawbacks of smoking of course"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(corpus)
M=tfidf.toarray()
#nmf = NMF(n_components=2,random_state=1)
#nmf.fit(tfidf)
print("M",M)
W,S=COSNMF(M,M.shape[0])
print("S",S)
print("W",W)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate((W,S)):
    print("topic #%d" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:10 - 1:-1]]))
    print()
    
""" for i, doc in enumerate(corpus):
    print("Document #%d" % i)
    print(doc)
    print("Topic distribution:", nmf.transform(tfidf[i])[0])
    print() """