# Twitter Sentiment Analysis (Naive Bayes + Gradio)

A beginner-friendly **binary Twitter sentiment classifier** built using **Python, NLTK, TF-IDF, scikit-learn, and Gradio**. It predicts whether text is **positive** or **negative**.

The training data is the [NLTK `twitter_samples` corpus](https://www.nltk.org/howto/twitter.html), which contains **5,000 positive** and **5,000 negative** labeled tweets. The corpus is downloaded automatically when you run the notebook or training script; it is **not** bundled in this repository.

## What this project demonstrates

- Cleaning text: lowercase conversion, URL and @mention removal, and hashtag-symbol removal (hashtag words are kept).
- Tweet-aware tokenization that preserves emoticons such as `:)` and `:(`.
- An 80/20 **stratified** train/test split with `random_state=42`.
- TF-IDF text features and a **Multinomial Naive Bayes** classifier.
- Accuracy, precision, recall, F1-score, a confusion matrix, and examination of errors.
- A Gradio app for trying your own tweets.

**Reported result from the original Colab session:** **95.9% test accuracy** (1,918/2,000 predictions correct). This figure comes from the original run and has not been independently reproduced in this ZIP. Retraining may give slightly different results across software versions.

> **Limitations:** The dataset is labeled positive/negative only; the model does not detect neutral sentiment. Emoticons in this corpus are strong shortcuts for the classifier, so 95.9% accuracy does **not** imply the same performance on new real-world tweets. This project does **not** scrape live X/Twitter data or automatically track sentiment by hashtag. The dataset is a public NLTK teaching corpus, not a live Twitter stream.

## Repository contents

```text
twitter-sentiment-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── sentiment_utils.py       # Cleaning and tokenization helpers
├── train.py                 # Rebuild, evaluate, and save the model
├── app.py                   # Gradio app
├── notebooks/
│   └── Twitter_Sentiment_Analysis.ipynb   # Self-contained Google Colab notebook
├── artifacts/
│   └── .gitkeep             # Model will be generated here
├── outputs/
│   └── .gitkeep             # Reports and confusion matrix generated here
└── tests/
    └── test_pipeline.py    # Basic automated tests
```

## Option A: Run in Google Colab (easiest)

1. Upload `notebooks/Twitter_Sentiment_Analysis.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Run all** (or execute each cell in order).
3. The notebook downloads the dataset, prepares the text, trains the model, evaluates it, and shows a confusion matrix.
4. Run the **Save model** cell to download `twitter_sentiment_model.pkl` to your computer.
5. The final **Gradio (optional)** cell can expose a temporary public share link. **Only run it if you want to share the app publicly.**

The Colab notebook is **standalone**: you do not need to upload other `.py` files to train or test it.

## Option B: Run locally (recommended for a GitHub clone)

Requires **Python 3.10+** and an internet connection **the first time** you download the NLTK corpus.

```bash
# Unzip the project, then open its directory
cd twitter-sentiment-analysis

# (Recommended) create a virtual environment
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Train and evaluate (downloads the NLTK corpus on first use)
python train.py

# Launch a local web interface
python app.py
```

Open the local URL printed by Gradio (usually `http://127.0.0.1:7860`). Enter a tweet and select **Submit**.

Training creates these files:

- `artifacts/twitter_sentiment_model.pkl` — trained model + TF-IDF vectorizer + tweet tokenizer + cleaning function.
- `outputs/metrics.json` — accuracy, precision, recall, and F1 scores.
- `outputs/classification_report.txt` — class-by-class evaluation report.
- `outputs/confusion_matrix.png` — confusion matrix chart.

**If you already saved the model in Colab:** place your own `twitter_sentiment_model.pkl` into the `artifacts/` directory, then run `python app.py` without retraining. That saved package should contain the keys `model`, `vectorizer`, `tokenizer`, and `clean_tweet` (as in the tutorial).

## Deploying to Hugging Face Spaces

1. Run `python train.py` locally **or** use your saved Colab `.pkl` file.
2. Create a new **Gradio** Space at [huggingface.co/spaces](https://huggingface.co/spaces).
3. Upload `app.py`, `sentiment_utils.py`, `requirements.txt`, and `artifacts/twitter_sentiment_model.pkl` into the same relative paths. Hugging Face Spaces may expect `app.py` at the root (as provided here).
4. The app runs with `demo.launch()` and does not need a `share=True` link on Spaces.

**Security:** A `.pkl` file can execute arbitrary code when unpickled. **Never load or deploy a pickle from an untrusted source.** Only use a model you created yourself or verified. In some environments, model files made with other versions of scikit-learn may need to be retrained.

## Try it

- `I love this amazing project :)` → likely positive
- `I hate this terrible day :(` → likely negative
- `I miss you :( ` → may be harder because the sentiment is contextual

These are example inputs; a probabilistic model can still make mistakes.

## Tests

```bash
python -m unittest discover -s tests -v
```

The unit tests do not download the Twitter dataset.

## Notes for GitHub

- The model pickle is **not included** in this ZIP because the original saved binary belongs to your Colab session. The training script/notebook regenerates it.
- Generated model, metrics, datasets, caches, and virtual environments are ignored by `.gitignore`. You can **deliberately** add your trained model for deployment if the hosting platform requires it.
- Avoid committing public Gradio temporary URLs; they expire.
- Add your own app screenshots to the README after running it, if desired.

## Tools

Python · NLTK · scikit-learn · TF-IDF · Multinomial Naive Bayes · matplotlib · cloudpickle · Gradio
