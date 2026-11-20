import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text)
    text = re.sub(r'#', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    tokens = re.findall(r"[a-z']+", clean(text))
    tokens = [t.strip("'") for t in tokens]
    return [t for t in tokens if len(t) >= 3 and t not in stop_words]