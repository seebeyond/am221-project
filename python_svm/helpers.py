import re
import Stemmer # PyStemmer - https://pypi.python.org/pypi/PyStemmer/1.0.1

# Function to extract n-grams
def ngrams(seq, n):
    "List all the (overlapping) ngrams in a sequence."
    return [' '.join(seq[i:i+n]) for i in range(1+len(seq)-n)]
    
# Function to clean text and return a list of words/n-grams
def process_text(text, stemmer):
    # get rid of escaped characters and apostrophes
    txt = text.lower().replace("&quot;","'").replace("'","")
    # reg-ex to split on
    words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
    words = filter(bool, words)
    # stemmer
    stm = stemmer.stemWords(words)
    # n-grams
#     words = stm + ngrams(words, 2)
    words = stm
    return words