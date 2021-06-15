import nltk
import re
import numpy as np
from nltk import sent_tokenize, word_tokenize, WordPunctTokenizer

# word punctuation tokenizer
wpt = WordPunctTokenizer()
# stop words
stop_words = nltk.corpus.stopwords.words('english')
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

normalize_corpus = np.vectorize(normalize_doc)