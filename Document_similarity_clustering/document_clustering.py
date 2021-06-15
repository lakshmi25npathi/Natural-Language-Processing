import nltk
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Building corpus of documents
corpus = ['The sky is blue and beautiful.',
 'Love this blue and beautiful sky!',
 'The quick brown fox jumps over the lazy dog.',
 "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
 'I love green eggs, ham, sausages and bacon!',
 'The brown fox is quick and the blue dog is lazy!',
 'The sky is very blue and the sky is very beautiful today',
 'The dog is lazy but the brown fox is quick!'
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals',
'weather', 'animals']

# convert corpus list to array 
corpus = np.array(corpus)
# Creating dataframe
corpus_df = pd.DataFrame({'Document': corpus,'Category': labels})
corpus_df = corpus_df[['Document','Category']]
print(corpus_df)

# Normalization of corpus
#Tokenization of each document
wpt = nltk.WordPunctTokenizer()
# Removing stop words
stop_words = nltk.corpus.stopwords.words('english')

# Normalization of each document
def normalize_doc(doc):
    # lowercase and remove special characters\whitespace
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

# convert corpus to vector form
norm_corpus = np.vectorize(normalize_doc)
# Apply function for text normalization
norm_corpus = norm_corpus(corpus)
print(norm_corpus)

#TF-IDF model for feature extraction
tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()
# Getting the feature names
vocab = tv.get_feature_names()
# Creating a dataframe
tv_df = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
print(tv_df)

# Cosine similarity to check similar documents
cs_matrix = cosine_similarity(tv_matrix)
cs_df = pd.DataFrame(cs_matrix)
print(cs_df)

# Document clustering with similarity features by using unsupervised agglomerative hierarchical clustering algorithm.
Z = linkage(cs_df, 'ward')
print(Z) # linkage matrix
# Creating a dataframe from document clusters
doc_cluster = pd.DataFrame(Z, columns=['Document/Cluster 1', 'Document/Cluster 2','Distance','Cluster Size'],dtype='object')
print(doc_cluster)

# Plotting the document clusters 
plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point or Cluster')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
plt.show()

# Getting the cluster lables
max_dist = 1.0
cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns= ['ClusterLabel'])
corpus_cluster = pd.concat([corpus_df, cluster_labels], axis=1)
print(corpus_cluster)


