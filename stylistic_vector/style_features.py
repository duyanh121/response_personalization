import re
import numpy as np
import string
import emoji
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SLANG_WORDS = {
    "lol", "lmao", "bruh", "bro", "omg", "wtf",
    "fuck", "shit", "damn", "haha", "hehe", "hihi"
}

LAUGHTER_WORDS = {"haha", "hehe", "hihi", "lol"}
FIRST_PERSON = {"i", "me", "my", "mine"}

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def extract_features(text: str) -> np.ndarray:
    text = text.strip()
    tokens = tokenize(text)

    n_tokens = len(tokens) + 1e-6
    n_chars = len(text) + 1e-6

    punct_count = sum(1 for c in text if c in string.punctuation)
    exclaim_count = text.count("!")
    question_count = text.count("?")
    caps_count = sum(1 for c in text if c.isupper())
    emoji_count = emoji.emoji_count(text)

    slang_count = sum(1 for t in tokens if t in SLANG_WORDS)
    laughter_count = sum(1 for t in tokens if t in LAUGHTER_WORDS)
    first_person_count = sum(1 for t in tokens if t in FIRST_PERSON)
    stopword_count = sum(1 for t in tokens if t in ENGLISH_STOP_WORDS)

    avg_token_len = sum(len(t) for t in tokens) / n_tokens

    return np.array([
        n_tokens,
        avg_token_len,
        punct_count / n_chars,
        exclaim_count / n_chars,
        question_count / n_chars,
        caps_count / n_chars,
        emoji_count / n_tokens,
        float(emoji_count > 0),
        slang_count / n_tokens,
        laughter_count / n_tokens,
        first_person_count / n_tokens,
        stopword_count / n_tokens,
    ])

def extract_features_batch(texts):
    return np.vstack([extract_features(t) for t in texts])
