# Bank Marketing Conversion Prediction

An end-to-end, production-shaped ML pipeline predicting whether a bank
telemarketing call will result in a term deposit subscription, built as a
practice project in applying real software engineering discipline (clean
code, testing, logging, custom exceptions, an API layer) to a data science
workflow, not just a notebook that happens to run.

## Why this dataset

The dataset (UCI Bank Marketing, `bank-additional-full.csv`) looks simple
but hides a genuine target-leakage trap: `duration` (the call length) is
almost perfectly correlated with the outcome, because a call that never
converts is, by definition, short. A model trained with `duration` included
reports misleadingly great accuracy and is useless in production, you
don't know how long the call will last *before* you decide whether to make
it. This project deliberately drops it and treats the resulting, much
harder, imbalanced classification problem (~89% "no" / ~11% "yes") as the
real task.

## Key engineering decisions

- **Leak-safe pipeline** - all imputation, scaling, and encoding live
  inside a single `sklearn.Pipeline`/`ColumnTransformer`, fit only on the
  training fold in every cross-validation split. No statistic is ever
  computed on data the model will later be evaluated against.
- **Imbalance handled explicitly, not accidentally** - `class_weight='balanced'`
  for linear/tree models, `scale_pos_weight` for XGBoost, and model
  comparison is done on **ROC-AUC**, not accuracy (a model predicting "no"
  every time would score ~89% accuracy and be worthless).
- **Model selection via stratified cross-validation**, not a single
  train/test split - five models (Logistic Regression, Random Forest,
  Gradient Boosting, AdaBoost, XGBoost) are compared on mean CV ROC-AUC,
  the winner is refit on the full training set, and only then evaluated
  once on a held-out test set.
- **`unknown` as a category, not a null** - most missing values are
  imputed, but `default` keeps `"unknown"` as its own category rather than
  guessing, since non-disclosure may itself correlate with the outcome.
- **Custom exceptions** (`DataCleaningError`, `ArtifactLoadError`,
  `PredictionError`) map to distinct, meaningful HTTP status codes in the
  API, instead of every failure collapsing into a generic 500.
- **Tested, not just run** - cleaning, pipeline construction, prediction,
  and the API's success/error paths all have pytest coverage, including
  deliberately forced failure paths via `monkeypatch`.

## Project structure

```
src/
├── data/
│   └── cleaning.py         # stateless cleaning steps (no fitting, no leakage risk)
├── workflows/
│   ├── pipeline.py         # build_preprocessor / build_model_pipeline
│            # split, CV model selection, refit, artifact saving
├── models/
│   |── train.py            # train models / save artifacts
│   |── predict.py          # load_artifacts / predict_new
│   └── evaluate.py         # evaluating the model
├── api/
│   └── main.py             # FastAPI service
└── exceptions.py           # DataCleaningError, ArtifactLoadError, PredictionError
utils/
└── log.py                  # centralized logging config
tests/
├── test_cleaning.py
├── test_pipeline.py
├── test_predict.py
└── test_api.py
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` with:
```
DATASET_PATH=data/bank-additional-full.csv
ARTIFACT_PATH=artifact/artifacts.joblib
LOG_PATH = logs/app.log
```

## Usage

**Train and save a model:**
```bash
python -m src.workflows.train
```
Cleans the data, cross-validates five models on ROC-AUC, refits the best
one on the full training set, evaluates it once on the held-out test set,
and saves `{model, encoder}` to `ARTIFACT_PATH`.

**Predict from Python directly:**
```bash
python -m src.models.predict
```

**Run the API:**
```bash
uvicorn src.api.main:app --reload
```
Interactive docs at `http://localhost:8000/docs`.

`POST /predict` — accepts the raw feature set (see `InputModel` in
`src/api/main.py` for the full schema) and returns:
```json
{ "prediction": 0, "probability": 0.18 }
```

| Status | Meaning |
|---|---|
| 200 | Prediction succeeded |
| 400 | Input passed validation but failed cleaning |
| 422 | Input failed schema/type/range validation |
| 500 | Unexpected failure during prediction |

## Testing

```bash
pytest
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, pytest, joblib

## Not yet done

Dockerfile, CI (GitHub Actions), and a proper decision-threshold /
interpretability (SHAP) pass are the natural next steps beyond this stage.
