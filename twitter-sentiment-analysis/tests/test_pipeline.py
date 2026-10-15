"""Small offline tests (the 10,000-tweet dataset is not required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sentiment_utils import clean_tweet, identity_tokens, preprocess, classify_text


class SentimentUtilsTests(unittest.TestCase):
    def test_text_cleaning_preserves_emoticons(self):
        text = '#FollowFriday @someone I LOVE this :) https://example.com/x'
        self.assertEqual(clean_tweet(text), 'followfriday i love this :)')

    def test_tokenize_tweet(self):
        tokens = preprocess('I feel sad :(')
        self.assertEqual(tokens, ['i', 'feel', 'sad', ':('])

    def test_prediction_pipeline(self):
        tokenizer = TweetTokenizer()
        training_texts = ['love great :)', 'happy nice :)', 'hate awful :(', 'sad bad :(']
        X = [preprocess(t, tokenizer) for t in training_texts]
        y = [1, 1, 0, 0]
        vectorizer = TfidfVectorizer(analyzer=identity_tokens, lowercase=False)
        transformed = vectorizer.fit_transform(X)
        model = MultinomialNB().fit(transformed, y)
        package = {
            'model': model,
            'vectorizer': vectorizer,
            'tokenizer': tokenizer,
            'clean_tweet': clean_tweet,
        }
        self.assertEqual(classify_text(package, 'I feel happy :)'), 'Positive :)')
        self.assertEqual(classify_text(package, 'awful and sad :('), 'Negative :(')
        self.assertEqual(classify_text(package, ' '), 'Please enter a tweet.')


if __name__ == '__main__':
    unittest.main()
