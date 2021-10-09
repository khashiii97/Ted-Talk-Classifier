import string
import nltk
from hazm import *

English_stop_words_level1 = ['an', 'and', 'for', 'that', 'the', 'with', 'he', 'in', 'can', 'from', 'a', 'to', 'of',
                             'it', 'talk', 'how', 'you', 'thi', 'about', 'we', 'what', 'on', 'as', 'hi', 'is', 'us',
                             'our', 'at']
English_stop_words_level2 = ['do', 'her', 'be', 'but', 'by', 'she', 'are']
Persian_stop_words_level1 = ['را', 'با', 'به', 'و', 'آن', 'که', 'از', 'این', 'بر', 'در', 'یا', 'تا']
Persian_stop_words_level2 = ['اس', 'کرد', 'ایر', 'دو',  'یک', 'سال', 'نیز', 'بود', 'شد', 'خود', 'برا', 'دارد']


def prepare_text(text, lang="en"):
    tokens = []
    if lang == "en":
        stemmer = nltk.stem.porter.PorterStemmer()
        tokens = nltk.word_tokenize(text)
        tokens = [stemmer.stem(t) for t in tokens]
        tokens = [tok for tok in tokens if tok not in string.punctuation]  # removing punctuations
        tokens = [t.lower() for t in tokens if t.isalpha()]  # removing punctuations
        tokens = [tok for tok in tokens if tok not in English_stop_words_level1 + English_stop_words_level2]
    elif lang == "fa":
        normalizer = Normalizer()
        punctuations = [']]', '[[', '[', ']', '؟', '!', '.', ',', '،', '?', ')', '(', ')', '(', '=', '==', '===', '✔',
                        '«', '»', '//www', 'http', '</ref>', '||', '<ref', '<ref>', 'name=', '|-', 'of', '–', '/', '‹‹',
                        '|', '-', '♦️', '♠️', '≈', '&', '←', '†', '‘', '—', '★', '≥',  '\n', '\'', '"', '\\', '//',
                        '</sup>', '<sup>', '<math>', '<math>', '*', '+', '<', '>', ';', '_', '$', '#', "ş", "٪", "",
                        '\u200d', '`']
        for p in punctuations:
            text = text.replace(p, ' ')
        text = normalizer.normalize(text)
        temp_tokens = word_tokenize(text)
        tokens = []
        for tok in temp_tokens:
            if tok in punctuations or tok in string.punctuation:
                continue
            tok = tok.replace('\u200c', '').replace(' ', '')
            tok = Stemmer().stem(tok)
            if tok in (Persian_stop_words_level1 + Persian_stop_words_level2):
                continue

            is_not_persian = True
            for c in tok:
                if '\u061F' <= c <= '\u065f':
                    is_not_persian = False
                    break
                if '\u066B' <= c <= '\u06dc':
                    is_not_persian = False
                    break
            if is_not_persian:
                continue
            tokens.append(tok)
    return tokens
