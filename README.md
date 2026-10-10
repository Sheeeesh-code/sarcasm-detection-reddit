# Sarcasm Detection on Reddit (SARC)

Binary sarcasm detection on Reddit comments with the balanced split of the SARC dataset. The repository compares
classical baselines (logistic regression, random forest, XGBoost on TF-IDF features) with fine-tuned DistilBERT
models that also see the parent comment and the subreddit.

Course project for CS3244 (NUS), Group 20.

## Results

Test set, balanced classes (50% sarcastic). Figures are those of the [project report](report/CS3244_Group20_Final_Report.pdf).
F1 is the F1 of the sarcastic class for the baselines and the macro-averaged F1 for the neural models.

| Model | Notebook | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic regression (TF-IDF) | `baseline_models/logisticRegression.ipynb` | 0.7198 | 0.7102 | 0.7911 |
| Random forest (TF-IDF + subreddit + punctuation) | `baseline_models/tree.ipynb` | 0.7127 | 0.7034 | 0.7827 |
| XGBoost (SVD-300 + subreddit + punctuation) | `baseline_models/tree.ipynb` | 0.7074 | 0.6968 | 0.7814 |
| DistilBERT, comment only | `neural_network/neuralNetwork_simple.ipynb` | 0.7658 | 0.7675 | 0.8492 |
| DistilBERT, comment + parent | `neural_network/neuralNetwork_parent.ipynb` | 0.7747 | 0.7746 | 0.8593 |
| **DistilBERT, comment + parent + subreddit** | `neural_network/neuralNetwork_parent_and_subreddit.ipynb` | **0.7774** | **0.7771** | **0.8604** |

Notes on reading the table:

- The baselines classify the comment on its own (plus subreddit and punctuation features for the tree models).
  In `logisticRegression.ipynb`, an ablation shows that concatenating the parent comment to a TF-IDF bag of words
  hurts the linear model, whereas the transformer, which reads the pair jointly, benefits from it.
- The neural notebooks drop the rows with a missing comment or parent (283,064 test rows instead of 283,332), so the
  test sets differ slightly.

## Report

The full write-up (data processing, models, ablations, comparison with other sarcasm corpora) is in
[`report/CS3244_Group20_Final_Report.pdf`](report/CS3244_Group20_Final_Report.pdf) (project title: *"Totally
Accurate" Sentiment Analysis*). Names (students and mentor) and student ID numbers are anonymised.

## Repository layout

```
data_processing.ipynb          loading, EDA, cleaning, text normalisation, export of the cleaned CSVs
baseline_models/
    logisticRegression.ipynb
    tree.ipynb                 random forest and XGBoost
    logistic_regression_model.joblib
    xgboost_model.joblib
neural_network/
    neuralNetwork_simple.ipynb                comment only
    neuralNetwork_parent.ipynb                comment + parent comment
    neuralNetwork_parent_and_subreddit.ipynb  comment + parent comment + subreddit embedding (best)
    checkpoints/               PyTorch weights (not in git, see below)
data/                          dataset files (not in git, see data/README.md)
report/                        final project report (PDF)
requirements.txt
```

## How to run

1. **Environment**
   ```bash
   pip install -r requirements.txt
   ```
2. **Data**: download the SARC balanced files and put them in `data/` (see [`data/README.md`](data/README.md)).
3. **Preprocessing**: run `data_processing.ipynb` from top to bottom. It writes `data/train_cleaned.csv` and
   `data/test_cleaned.csv`, which every other notebook reads.
4. **Models**: run the notebooks in `baseline_models/` and `neural_network/` from their own folder (they read
   `../data/...`). The tokenizer and the DistilBERT weights are downloaded from the Hugging Face Hub on first use
   (public model, no token needed). Training the neural models takes several hours on a GPU; to only evaluate, skip
   the training cell and load the checkpoint.

## Large files

GitHub rejects files above 100 MB, so the following are not stored in git:

| File | Where to put it |
|---|---|
| `data/train-balanced.csv.bz2`, `data/test-balanced.csv.bz2` | `data/` (raw SARC) |
| `data/train_cleaned.csv`, `data/test_cleaned.csv` | `data/` (produced by `data_processing.ipynb`) |
| `baseline_models/random_forest_model.joblib` (about 460 MB) | `baseline_models/` |
| `neural_network/checkpoints/*.pth` | `neural_network/checkpoints/` |

Trained files: _add the download link here once they are uploaded (GitHub release or Hugging Face Hub)._

The `.joblib` files are Python pickles: only load files you trust, with the library versions in `requirements.txt`.

## Dataset

SARC (Self-Annotated Reddit Corpus), balanced split: M. Khodak, N. Saunshi, K. Vodrahalli, *A Large Self-Annotated
Corpus for Sarcasm*, LREC 2018. The data are Reddit comments and are not redistributed here.
