# Spam Email Detection

A text classification model that detects whether an email is spam or legitimate ("ham"), built and compared using Naive Bayes and Logistic Regression.

## Project Overview

This project walks through a full NLP classification workflow:

- Loading and exploring a labeled spam/ham email dataset
- Cleaning and preprocessing raw email text
- Converting text into numeric features using TF-IDF vectorization
- Training and evaluating two classification models (Naive Bayes and Logistic Regression)
- Comparing model performance using precision, recall, and F1-score
- Saving the best-performing model pipeline for reuse
- Testing the saved pipeline on new, unseen example emails

## Dataset

The dataset (`spam_ham_dataset.csv`) contains 5,171 labeled emails:

| Column | Description |
|---|---|
| `label` | Text label: `ham` or `spam` |
| `text` | Raw email content |
| `label_num` | Numeric label (0 = ham, 1 = spam) |

**Class distribution:** 3,672 ham (71%) vs 1,499 spam (29%) — moderately imbalanced, addressed using stratified train/test splitting.

## Methodology

1. **Data cleaning** — lowercased text, removed the repeated "Subject:" prefix, stripped line breaks and punctuation, and collapsed extra whitespace.
2. **Train/test split** — 80/20 split, stratified by label to preserve class balance in both sets.
3. **Vectorization** — text converted into numeric features using `TfidfVectorizer` (English stopwords removed, vocabulary capped at 5,000 features).
4. **Model training** — two classifiers were trained and compared:
   - Multinomial Naive Bayes
   - Logistic Regression
5. **Evaluation** — precision, recall, F1-score, and confusion matrices were used to assess performance (accuracy alone can be misleading on imbalanced data).
6. **Pipeline saving** — the best-performing model, bundled together with its TF-IDF vectorizer, was saved with `joblib` for reuse on new text.

## Results

| Metric | Naive Bayes | Logistic Regression |
|---|---|---|
| Accuracy | 95% | **98%** |
| Ham F1-score | 0.96 | **0.99** |
| Spam F1-score | 0.91 | **0.97** |

**Logistic Regression was selected as the final model**, outperforming Naive Bayes across every metric — most notably on spam detection, the harder of the two classes.

### Example Predictions

The saved pipeline was tested on new, unseen messages. It reliably caught clear promotional/scam content (vouchers, urgent merchant notices, unsolicited offers) and correctly passed genuine work correspondence as ham. It was less reliable on casual, personal-style messages, since the training data consists mainly of business and promotional email content — a known limitation discussed further in the notebook.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (TF-IDF vectorization, modeling, evaluation, pipelines)
- joblib (model persistence)

## Project Structure

```
├── spam_ham_dataset.csv                        # Dataset
├── Spam_Detector.ipynb                         # Full analysis and modeling notebook
├── spam_detector_pipeline_naivebayers.pkl      # Saved Naive Bayes pipeline (vectorizer + model)
├── spam_detector_pipeline_logreg.pkl           # Saved final pipeline (vectorizer + Logistic Regression)
└── README.md
```

> Both trained pipelines are saved to disk. The Naive Bayes pipeline (`spam_detector_pipeline_naivebayers.pkl`) is kept for comparison, while the Logistic Regression pipeline (`spam_detector_pipeline_logreg.pkl`) is the recommended model for actual use, based on its stronger evaluation metrics below.

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
2. Open `Spam_Detector.ipynb` in Jupyter or VS Code.
3. Run all cells in order from top to bottom.

## Using the Saved Pipeline on New Text

```python
import joblib

# Recommended: Logistic Regression pipeline (best performance)
pipeline = joblib.load('spam_detector_pipeline_logreg.pkl')

# Alternative: Naive Bayes pipeline
# pipeline = joblib.load('spam_detector_pipeline_naivebayers.pkl')

# Note: apply the same clean_text() preprocessing used during training
# before passing new text into the pipeline.
prediction = pipeline.predict([cleaned_text])
```

## Future Improvements

- Expand training data to include more diverse, informal message styles
- Test additional models (e.g. Support Vector Machines)
- Hyperparameter tuning for both vectorizer and classifier
- Address remaining class imbalance with techniques like class weighting or resampling
