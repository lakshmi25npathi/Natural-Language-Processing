import re
import bs4
from bs4 import BeautifulSoup
import unicodedata
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer

# Removing html tags
def remove_html_tags(text_data):
    soup = BeautifulSoup(text_data, "html.parser")
    text = soup.get_text()
    return text

# removing accented characters
def remove_accent_chars(text_data):
    text_data = unicodedata.normalize('NFKD', text_data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text_data

# Text tokenization
def tokenize_text(text_data):
    sentences = nltk.sent_tokenize(text_data)
    token_words = [nltk.word_tokenize(sentence) for sentence in sentences]
    return token_words

# remove special characters
def remove_special_chars(text_data, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text_data = re.sub(pattern, '', text_data)
    return text_data

# remove repeated characters
def remove_repeated_chars(tokens):
   pattern = re.compile(r'(\w*)(\w)\2(\w*)')
   match_substitution = r'\1\2\3'

   def replace(old_w):  ## old word
        if wordnet.synsets(old_w):
            return old_w
        new_w = pattern.sub(match_substitution, old_w) # new word
        return replace(new_w) if new_w != old_w else new_w
        correct_tokens = [replace(w) for w in tokens]  #word
        return correct_tokens

# text stemming
def text_stemmer(text_data):
    ps = PorterStemmer()
    text_data = ' '.join([ps.stem(w) for w in text_data.split()])
    return text_data
    
# text lemmatization
wnl = WordNetLemmatizer()
# Using spacy library
nlp = spacy.load('', parse=True, tag=True, entity=True)
def text_lemmatizer(text_data):
    text_data = nlp(text_data)
    text_data = ' '.join([w.lemma_ if w.lemma_ != '-PRON-' else w.text
    for w in text_data])
    return text_data

# remove stop words
tokenizer = ToktokTokenizer()
stopwords_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text_data, is_lower_case=False):
    tokens = tokenizer.tokenize(text_data)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in
        stopwords_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not
    in stopwords_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
