
import re, unicodedata,string
import nltk 
from bs4 import BeautifulSoup
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer


class text_processing():


    def __init__(self,text):
        self.text = text
     
    def remove_html_strips(self):
        soup = BeautifulSoup(self.text,"html.parser")
        self.text = soup.get_text()
        return self
    
    def remove_square_brackets(self):
        self.text = re.sub('\[^]*\]', '', str(self.text))
        return self
    
    def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)
        return self
    
    def tokenize_text(self):
        self.tokens = word_tokenize(self.text)
        return self
    
    def to_lowercase(self):
        """ Convert all characters to lower case from list of tokenized tokens"""
        new_tokens = []
        for token in self.tokens:
            new_token = token.lower()
            new_tokens.append(new_token)

        self.tokens = new_tokens
        return self
    
    def stem_tokens(self):
        """ stem tokens in list of tokenized tokens"""
        stemmer = PorterStemmer()
        stems = []
        for token in self.tokens:
            stem = stemmer.stem(token)
            stems.append(stem)
        self.tokens = stems
        return self
    
    def lemma_tokens(self):
        """ lemma tokens in list of tokenized tokens"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for token in self.tokens:
            lemma = lemmatizer.lemmatize(token)
            lemmas.append(lemma)
        self.tokens = lemmas
        return self
    
    def remove_stopwords(self):
        """Removing the stopwords from the list of tokenized words"""
        new_words = []
        for word in self.tokens:
            if word not in stopwords.words('english'):
                new_words.append(word)
        self.tokens = new_words
        return self
    
    def join_words(self):
        self.tokens = ' '.join(self.tokens)
        return self
    
    def do_all(self,text):

        self.text = text
        self = self.remove_html_strips()
        self = self.remove_square_brackets()
        self = self.remove_numbers()
        self = self.tokenize_text()
        self = self.to_lowercase()
        self = self.stem_tokens()
        self = self.lemma_tokens()
        self = self.remove_stopwords()
        self = self.join_words()

        return self.tokens 
