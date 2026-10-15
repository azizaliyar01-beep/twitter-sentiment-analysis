"""Gradio web app for the trained Twitter sentiment classifier.

First run `python train.py`, or place the trusted Colab pickle at
`artifacts/twitter_sentiment_model.pkl`, then run `python app.py`.
"""

from functools import lru_cache
from pathlib import Path

import cloudpickle
import gradio as gr

from sentiment_utils import classify_text

MODEL_PATH = Path(__file__).resolve().parent / 'artifacts' / 'twitter_sentiment_model.pkl'


@lru_cache(maxsize=1)
def load_package():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f'Model file not found at {MODEL_PATH}. '
            'Run `python train.py` first or copy your trusted Colab model '
            'to artifacts/twitter_sentiment_model.pkl.'
        )
    # WARNING: never unpickle model files from untrusted sources.
    with MODEL_PATH.open('rb') as f:
        package = cloudpickle.load(f)
    required = {'model', 'vectorizer', 'tokenizer', 'clean_tweet'}
    if not required.issubset(package):
        raise ValueError('Saved model is missing required components.')
    return package


def predict_sentiment(tweet):
    if not tweet or not tweet.strip():
        return 'Please enter a tweet.'
    try:
        package = load_package()
    except FileNotFoundError as exc:
        return str(exc)
    return classify_text(package, tweet)


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3, placeholder='Type a tweet here...', label='Tweet'
    ),
    outputs=gr.Textbox(label='Predicted sentiment'),
    title='Twitter Sentiment Analysis',
    description=(
        'Positive vs. negative sentiment using NLTK, TF-IDF, '
        'and Multinomial Naive Bayes. No neutral class.'
    ),
    examples=[
        ['I love this amazing project :)'],
        ['I hate this terrible day :('],
        ['This was such a fun experience!'],
    ],
)


if __name__ == '__main__':
    demo.launch()  # Local by default; don't expose a public link unintentionally.
