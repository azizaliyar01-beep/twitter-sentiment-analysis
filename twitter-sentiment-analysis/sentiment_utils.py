"""Helpers used by the sentiment training script and app."""

import re
from nltk.tokenize import TweetTokenizer


def clean_tweet(tweet: str) -> str:
    """Normalize tweets without removing sentiment-bearing emoticons."""
    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


def identity_tokens(tokens):
    """Pass pre-tokenized text through unchanged to TF-IDF."""
    return tokens


def preprocess(tweet: str, tokenizer: TweetTokenizer | None = None):
    """Return a list of tweet tokens; retain :) and :( emoticons."""
    tokenizer = tokenizer or TweetTokenizer()
    return tokenizer.tokenize(clean_tweet(tweet))


def classify_text(model_package: dict, tweet: str) -> str:
    """Return Positive or Negative using a saved training package."""
    if not isinstance(tweet, str) or not tweet.strip():
        return 'Please enter a tweet.'

    cleaner = model_package.get('clean_tweet', clean_tweet)
    tokenizer = model_package['tokenizer']
    tokens = tokenizer.tokenize(cleaner(tweet))
    features = model_package['vectorizer'].transform([tokens])
    prediction = int(model_package['model'].predict(features)[0])
    return 'Positive :)' if prediction == 1 else 'Negative :('
