from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import nltk
import json
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenization(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token.isalnum() and token not in stop_words and not token.isnumeric()
    ]
    return filtered_tokens

def getNgrams(text):
    tokens = tokenization(text)
    unigrams = list(ngrams(tokens, 1))
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    all_ngrams = unigrams + bigrams + trigrams
    ngram_counts = Counter(all_ngrams)
    return ngram_counts

def classify(content_dict):
    classified_pages = {}
    for page, page_data in content_dict.items():
        text = page_data['text']
        top_ngrams = getNgrams(text)
        top_topics = list(dict(top_ngrams.most_common(20)).keys())
        flat_topics = [' '.join(topic) for topic in top_topics]
        classified_pages[page] = flat_topics
    return classified_pages

def plot(ngram_counts, page_no):
    ngrams, counts = zip(*ngram_counts.most_common(20))
    ngrams = [' '.join(ngram) for ngram in ngrams]
    plt.figure(figsize=(20, 6))
    plt.barh(ngrams, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.title(f'Top 20 n-grams for {page_no}')
    plt.gca().invert_yaxis()
    # plt.savefig(f'{page_no}_ngrams.png')
    plt.close()

def index(path):
    reader = PdfReader(path)
    content_dict = {}

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        page_no = f'Page {i + 1}'
        content_dict[page_no] = {'text': text}

    classified_pages = classify(content_dict)

    with open('content_dict.json', 'w') as f:
        json.dump(content_dict, f)

    with open('page_classification.json', 'w') as f:
        json.dump(classified_pages, f)

    for page, page_data in content_dict.items():
        text = page_data['text']
        ngram_counts = getNgrams(text)
        plot(ngram_counts, page)

index('Dsa.pdf')
