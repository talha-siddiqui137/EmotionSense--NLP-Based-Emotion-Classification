import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Clean raw text so it is ready for TF-IDF.
    Steps: lowercase -> remove punctuation -> remove numbers ->
    remove non-ascii -> tokenize -> remove stopwords -> join back to string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = text.encode("ascii", "ignore").decode()

    words = word_tokenize(text)
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)
