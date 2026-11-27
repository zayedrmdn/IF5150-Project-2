# IF5150 Project 2

Machine learning algorithm implementation for IF5150 Advanced Artificial Intelligence. The task is to classify each student's academic outcome (Dropout, Enrolled or Graduate).

## Description

- `src/` holds the from-scratch models, written with numpy only: ID3 (`id3.py`), Random Forest (`random_forest.py`), KNN (`knn.py`), Logistic Regression with One-vs-Rest (`logistic_regression.py`) and an ANN (`neural_network.py`), plus a hard-voting ensemble of all five (`ensemble.py`). `evaluate.py` computes the metrics shared by all models. Every model has `fit`, `predict`, `save` and `load`.
- `data/` holds the preprocessed train, validation and test splits (60:20:20) and the preprocessed Kaggle test set.
- `notebooks/` holds the experiments:
  - `sklearn_dt_rf.ipynb`: tuning ID3 and Random Forest on the validation split, the scikit-learn counterparts, save and load.
  - `sklearn_knn.ipynb`: KNN and its scikit-learn counterpart.
  - `sklearn_logreg_ann.ipynb`: Logistic Regression and ANN, their scikit-learn counterparts, save and load.
  - `comparison2.ipynb`: all models on the test split, including the from-scratch ensemble and the scikit-learn VotingClassifier, the comparison table, the report figures and the Kaggle submission file. The report uses this notebook.
  - `comparison.ipynb`: the earlier version of the comparison, without the from-scratch ensemble.
- `doc/` holds the report (PDF) and its figures.
- `submissions/` holds the Kaggle submission file.

## Setup and run

Requires Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS or Linux)
pip install -r requirements.txt
pip install notebook
```

Open the notebooks from the `notebooks/` folder with `jupyter notebook`, then run all cells. Run `sklearn_dt_rf.ipynb` and `sklearn_logreg_ann.ipynb` first; `comparison2.ipynb` uses the hyperparameters chosen there. Trained models are written to `models/` and figures to `doc/figures/`.

## Task division

| Member | NIM | Tasks |
|---|---|---|
| Zayed Ramadan Rahmat | 23526015 | ID3, Random Forest, scikit-learn VotingClassifier, comparison and Kaggle submission, shared `evaluate()`, repository and README |
| Ibnu Fikri | 23526032 | Data cleaning and preprocessing, KNN, ensemble |
| Rabab Fawaz Saleh Alali | 23526701 | Logistic Regression, ANN |
