import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Download necessary NLTK data (if not downloaded already)

text = "NLTK is awesome!"
tokens = word_tokenize(text)
print(tokens)
