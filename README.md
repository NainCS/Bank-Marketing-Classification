# Bank Marketing Classification — Predicting Term Deposit Subscriptions on Imbalanced Data

Predicting whether a bank client will subscribe to a term deposit after a telemarketing call, using the UCI Bank Marketing dataset. Only about **12% of clients said yes**, so the main challenge is class imbalance: a model can reach ~90% accuracy by predicting "no" for almost everyone while missing most real subscribers.

The project compares preprocessing strategies and three algorithm families (linear, bagging, boosting), then benchmarks the results against published work on the same data. It was reproduced in both **Python** and **Weka**.

> Built for the Data Mining and Machine Learning module, BSc (Hons) Data Science, National College of Ireland.

---

## Why accuracy is not enough

The untuned baseline XGBoost model reached **90.5% accuracy** on the full dataset. Its confusion matrix on the 9,043-row test set tells a different story:

|                 | Predicted no | Predicted yes |
| --------------- | ------------ | ------------- |
| **Actual no**   | 7,686        | 299           |
| **Actual yes**  | 556          | 502           |

It found only **502 of the 1,058 real subscribers** (recall 0.47). For a bank, those 556 missed customers are the whole point of the model. That is why this project evaluates with **precision, recall, F1-score and ROC-AUC**, not accuracy alone.

---

## Approach

The project follows the **KDD (Knowledge Discovery in Databases)** process in two stages.

### Stage 1 — Choose the preprocessing strategy (full dataset, 45,211 rows)

Five preprocessing scenarios were tested with Logistic Regression, Random Forest and XGBoost on an 80/20 stratified split:

| Scenario | What it does | Best F1 | Decision |
| --- | --- | --- | --- |
| Baseline | One-hot encoding + scaling, no resampling | 0.540 (XGB) | Discarded |
| IQR capping | Caps numeric outliers | 0.529 (XGB) | Discarded |
| **One-hot + SMOTE** | **Creates synthetic minority examples** | **0.607 (XGB)** | **Selected** |
| SMOTENC | SMOTE variant for mixed categorical/numeric data | 0.586 (XGB) | Discarded |
| IQR capping + SMOTE | Both combined | 0.578 (XGB) | Discarded |

One-hot encoding + SMOTE was the most consistent across all three algorithms. With it, XGBoost's recall rose from 0.47 to 0.61 and F1 from 0.54 to 0.61.

Interestingly, capping outliers made results slightly worse, which suggests the extreme values carry useful information rather than noise.

### Stage 2 — Tune and benchmark (10% subsample, 4,521 rows)

The selected pipeline was applied to the official 10% subsample so results could be compared with previous studies on the same data. Each model was tuned with **stratified 10-fold cross-validation**:

| Algorithm | Search method | Parameters tuned |
| --- | --- | --- |
| Logistic Regression | GridSearchCV | solver, C (regularisation), class_weight |
| Random Forest | GridSearchCV | n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf |
| XGBoost | RandomizedSearchCV (larger search space) | n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda |

### Avoiding data leakage

Imputation, scaling, encoding and SMOTE all run **inside an `imblearn` Pipeline**. This means SMOTE only creates synthetic rows from each fold's training data, never from the rows used for testing. Applying SMOTE before splitting would let synthetic copies of test rows leak into training and inflate the scores.

---

## Results (10% subsample, 10-fold CV, "yes" class)

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression + SMOTE (tuned) | 0.827 | 0.378 | **0.789** | 0.511 |
| **Random Forest + SMOTE (tuned)** | **0.891** | **0.520** | 0.630 | **0.569** |
| XGBoost + SMOTE (tuned) | 0.888 | 0.509 | 0.610 | 0.554 |

- **Random Forest** gave the best overall balance on the subsample. Tuning raised its F1 from 0.44 to 0.57 and its recall from 0.37 to 0.63.
- **Logistic Regression** found the most subscribers (highest recall) but with many more false alarms (lowest precision).
- On the **full dataset**, XGBoost performed best, so the strongest model depends on the amount of data and the evaluation setup.

### Comparison with published work

Vitório & Marques (2021) used the same 10% subsample with 10-fold CV:

| Study | Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Vitório & Marques | Neural network, imbalanced data | 0.907 | 0.643 | 0.455 | 0.533 |
| Vitório & Marques | Random Forest, balanced by undersampling | 0.813 | 0.802 | 0.831 | 0.816 |
| This project | Random Forest + SMOTE (tuned) | 0.891 | 0.520 | 0.630 | 0.569 |

Compared with their imbalanced-data model, this project finds considerably more subscribers (recall 0.63 vs 0.46) with a higher F1. Their balanced Random Forest scores higher, but it was built by **undersampling**, which throws away majority-class rows. The two results are not measured on the same class distribution, so they are not directly comparable. SMOTE was chosen here to keep all the original data.

### Weka replication

The Python pipelines were rebuilt in Weka using `MultiFilter` (standardise → one-hot → SMOTE) and the `ScikitLearnClassifier` wrapper with the tuned hyperparameters:

| Model | Python F1 | Weka F1 |
| --- | --- | --- |
| Logistic Regression | 0.511 | 0.515 |
| Random Forest | 0.569 | 0.541 |
| XGBoost | 0.554 | 0.547 |

Logistic Regression was reproduced almost exactly, and the tree models closely, showing the results are reproducible outside the original code.

---

## Limitations

- **The `duration` feature leaks the outcome.** It records how long the call lasted, which is only known after the call has ended, when the result is already known. The dataset documentation recommends removing it for realistic prediction. The results above include it, so they are optimistic. *(Planned: rerun without `duration` and compare.)*
- **Tuning and evaluation used the same CV folds**, which makes the reported scores slightly optimistic. Nested cross-validation would give a stricter estimate.
- Many categorical columns contain `"unknown"` values. Treating them as their own category versus imputing the most frequent value made little difference, so they were kept as they are.

## Future work

- Adjust the decision threshold and calibrate probabilities to trade precision against recall for the minority class.
- Try cost-sensitive learning, since missing a subscriber and making an unnecessary call have different business costs.
- Test the `bank-additional` dataset, which adds five macroeconomic indicators.

---

## Repository structure

```
.
├── Datasets/
│   ├── bank/                          original UCI files (bank-full.csv, bank.csv)
│   ├── bank-additional/               extended dataset (not used in final models)
│   └── Split datasets/                train/test splits exported for Weka
├── Notebooks/
│   ├── Own Comparison of Full Dataset/        Stage 1: preprocessing scenarios
│   │   ├── Baseline_Model.ipynb
│   │   ├── Outlier_capping.ipynb
│   │   ├── SMOTE.ipynb
│   │   ├── SMOTENC.ipynb
│   │   └── Outliers+SMOTE.ipynb
│   └── Subset comparison against papers/      Stage 2: tuning and benchmark
│       ├── LR_Sample_Comparison.ipynb
│       ├── RF_Sample_Comparison.ipynb
│       ├── XGBoost_Sample_Comparison.ipynb
│       └── plots.ipynb
├── Weka/                              Weka run outputs for each model
├── pyproject.toml
└── requirements.txt
```

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · Matplotlib · Jupyter · Weka

## How to run

```bash
git clone https://github.com/NainCS/bank-marketing-classification.git
cd bank-marketing-classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

Open any notebook in `Notebooks/` and run the cells in order. The data is already included in `Datasets/`, and all notebooks use relative paths, so no changes are needed.

> **Note:** Run `LR_Sample_Comparison`, `RF_Sample_Comparison` and `XGBoost_Sample_Comparison` before `plots.ipynb`, since they create the CSV files it uses.

## Data source

S. Moro, P. Rita and P. Cortez, *Bank Marketing*, UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5K306

Based on: S. Moro, P. Cortez and P. Rita, "A data-driven approach to predict the success of bank telemarketing," *Decision Support Systems*, vol. 62, pp. 22–31, 2014.

Benchmark study: A. Vitório and G. Marques, "Impact of Imbalanced Data on Bank Telemarketing Calls Outcome Forecasting using Machine Learning," *ICDABI 2021*, pp. 380–384.
