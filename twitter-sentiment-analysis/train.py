"""Train and evaluate the Twitter sentiment model using NLTK samples.

Run: python train.py
"""

import json
from pathlib import Path

import cloudpickle
import matplotlib
matplotlib.use('Agg')  # Save figures on servers without a display
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sentiment_utils import clean_tweet, identity_tokens

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
OUTPUT_DIR = BASE_DIR / 'outputs'


def load_corpus():
    """Download the NLTK corpus on first use; needs internet once."""
    try:
        positive = twitter_samples.strings('positive_tweets.json')
        negative = twitter_samples.strings('negative_tweets.json')
    except LookupError:
        print('NLTK twitter_samples not found: downloading dataset...')
        if not nltk.download('twitter_samples', quiet=True):
            raise RuntimeError(
                'Dataset download failed. Check internet connectivity, '
                'then rerun python train.py.'
            )
        positive = twitter_samples.strings('positive_tweets.json')
        negative = twitter_samples.strings('negative_tweets.json')
    return positive, negative


def train():
    positive, negative = load_corpus()
    print(f'Dataset: {len(positive)} positive + {len(negative)} negative tweets')

    tokenizer = TweetTokenizer()
    X = [tokenizer.tokenize(clean_tweet(tweet)) for tweet in positive + negative]
    y = [1] * len(positive) + [0] * len(negative)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(analyzer=identity_tokens, lowercase=False)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    print(f'Training tweets: {len(X_train)} | Testing tweets: {len(X_test)}')
    print(f'TF-IDF features: {X_train_tfidf.shape[1]}')

    labels = ['Negative', 'Positive']
    report = classification_report(y_test, y_pred, target_names=labels)
    print('\n' + report)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    result = {
        'dataset': 'NLTK twitter_samples',
        'training_samples': len(X_train),
        'testing_samples': len(X_test),
        'features': X_train_tfidf.shape[1],
        'accuracy': round(float(accuracy), 6),
        'classes': {
            label: {
                'precision': round(float(precisions[i]), 6),
                'recall': round(float(recalls[i]), 6),
                'f1_score': round(float(f1s[i]), 6),
                'support': int(supports[i]),
            } for i, label in enumerate(labels)
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    package = {
        'model': model,
        'vectorizer': vectorizer,
        'tokenizer': tokenizer,
        'clean_tweet': clean_tweet,
    }
    with (ARTIFACT_DIR / 'twitter_sentiment_model.pkl').open('wb') as f:
        cloudpickle.dump(package, f)

    (OUTPUT_DIR / 'metrics.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8'
    )
    (OUTPUT_DIR / 'classification_report.txt').write_text(
        report + '\n', encoding='utf-8'
    )

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels
    ).plot(cmap='Blues', values_format='d')
    plt.title('Twitter Sentiment Analysis — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=160)
    plt.close()

    print('\nSaved:')
    print(' - artifacts/twitter_sentiment_model.pkl')
    print(' - outputs/metrics.json')
    print(' - outputs/classification_report.txt')
    print(' - outputs/confusion_matrix.png')
    print('\nFive example errors:')
    wrong = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    for i in wrong[:5]:
        print(f' - Actual={labels[y_test[i]]}, Predicted={labels[y_pred[i]]}: '
              f'{" ".join(X_test[i])[:140]}')


if __name__ == '__main__':
    train()
