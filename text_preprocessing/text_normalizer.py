import text_preprocessing
from text_preprocessing import remove_html_tags
from text_preprocessing import remove_accent_chars
from text_preprocessing import text_lemmatizer,text_stemmer
from text_preprocessing import remove_special_chars
from text_preprocessing import remove_stopwords
from text_preprocessing import tokenize_text

def normalize_text(text, html_removing=True, accent_char_remove=True, 
                   text_lower_case=True, lemmatize_text=True, special_char_remove=True, 
                   stop_words_removal=True, remove_digits=True, tokenize_text=True):

    normalize_text = []
    # normalize each document in the text

    for doc in text:
        # strip html
        if html_removing:
            doc = remove_html_tags(doc)
        # remove accented characters
        if accent_char_remove:
            doc = remove_accent_chars(doc)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # lemmatize text 
        if lemmatize_text:
            doc = text_lemmatizer(doc)
        # remove special characters or digits
        if special_char_remove: 
            doc = remove_special_chars(doc, remove_digits=remove_digits)
        # remove stopwords
        if stop_words_removal:
            doc = stop_words(doc, is_lower_case=text_lower_case)
        if tokenize_text:
            doc = tokenize_text(doc)
    
        normalize_text.append(doc)
    
        return normalize_text

