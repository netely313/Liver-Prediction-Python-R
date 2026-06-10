# Liver Disease Prediction with Logistic Regression

Binary classification of liver disease from routine clinical blood markers, built end-to-end with a leakage-aware machine learning pipeline.

This is a data-analysis portfolio project. It walks through the full workflow: data cleaning, exploratory analysis, careful train/test handling, model training with cross-validation, threshold tuning, and an honest evaluation of the results.

## Project Overview

Liver disease is often detectable from standard blood panels (bilirubin, liver enzymes, proteins) before symptoms become severe. The goal here is to predict whether a patient has liver disease (`Selector` = 1) or not (`Selector` = 0) using these affordable, widely available blood markers, and to understand *which* markers drive the prediction.

The emphasis is not just on getting a score, but on doing the modeling correctly (no data leakage) and reporting results transparently, including the model's limitations.

## Dataset

- **Records:** 583 raw rows, reduced to **570** after removing **13 duplicate** rows (duplicates can leak identical patients across train/test and inflate metrics).
- **Missing values:** none.
- **Features (11 columns):** `Age`, `Gender`, `TB` (total bilirubin), `DB` (direct bilirubin), `Alkphos` (alkaline phosphatase), `Sgpt` (ALT), `Sgot` (AST), `TP` (total protein), `ALB` (albumin), `A/G Ratio` (albumin/globulin ratio).
- **Target:** `Selector` — Liver Disease (1) vs No Liver Disease (0).
- **Class balance:** imbalanced — **71.2%** Liver Disease vs **28.8%** No Liver Disease.

## Methodology

The pipeline is designed to avoid data leakage at every step:

- **Duplicate removal** before any analysis.
- **No global outlier removal.** Elevated bilirubin and liver enzymes are clinically meaningful disease signals, not noise. Removing them would drop the sickest patients and (if done before splitting) leak information.
- **Train/test split first** — 80/20, **stratified** on the target to preserve class proportions (train: 456, test: 114).
- **Feature selection on the training set only.** `DB` and `TP` are dropped due to multicollinearity (with `TB` and `ALB` respectively), leaving **8 model features**.
- **Modeling pipeline:** `StandardScaler` + `LogisticRegression(class_weight='balanced')`, wrapped in a scikit-learn `Pipeline` so scaling is refit inside each CV fold.
- **5-fold Stratified cross-validation** on the training data.
- **Decision threshold tuned** on out-of-fold (cross-validated) training probabilities — never on the test set. The threshold maximizing F1 on train CV was **0.2**.
- **Final evaluation performed once** on the held-out test set.

## Results

Cross-validation (training set):

- ROC-AUC: **0.717**
- Balanced accuracy: 0.670
- Accuracy: 0.621

Held-out test set (n = 114):

- **ROC-AUC: 0.800**
- Accuracy: 0.711
- Precision: 0.714
- Recall: 0.988
- F1-score: 0.829
- Balanced accuracy: 0.509

Confusion matrix (test set):

- True Negatives: 1
- False Positives: 32
- False Negatives: 1
- True Positives: 80

### Honest interpretation

The **ROC-AUC of 0.80** shows the model's predicted probabilities genuinely rank patients well — sick patients tend to get higher scores than healthy ones.

However, the F1-maximizing threshold (0.2) pushes the model to label almost everyone as "disease": recall is 0.99 but **balanced accuracy is ~0.51, essentially chance** at distinguishing the two classes. On an imbalanced dataset where 71% of patients are positive, optimizing F1 rewards this "predict positive" behavior. The model catches nearly all true cases but barely identifies healthy patients (only 1 of 33 correctly).

This is the classic precision/recall and threshold trade-off in imbalanced clinical screening: a high-recall model is useful as a first-pass screen, but a different threshold (e.g. 0.4–0.5) would trade recall for meaningfully better balanced accuracy if false positives are costly.

## Feature Effects (Odds Ratios)

Odds ratios from the logistic model, expressed per 1 standard deviation increase (features are standardized). Values above 1 increase the odds of liver disease; below 1 are protective.

- **TB (total bilirubin): 3.48** — strongest risk driver
- **Sgpt / ALT: 3.32**
- **Sgot / AST: 2.93**
- Age: 1.36
- Alkphos (alkaline phosphatase): 1.21
- ALB (albumin): 1.16
- Gender (Male = 1): 1.14
- **A/G Ratio: 0.74** — protective

These align with clinical intuition: bilirubin and the liver transaminases (ALT, AST) are the dominant indicators of liver disease.

## R Analysis: Logistic Regression (Explanatory Model)

As a complementary analysis, I fit a logistic regression in R (`glm`, binomial family) on the deduplicated dataset (570 records, after removing 13 duplicate rows — consistent with the Python pipeline) to interpret which blood markers are statistically associated with liver disease. Unlike the Python pipeline (focused on prediction with a held-out test set), this R model is **explanatory** — its goal is inference on feature effects.

The target was encoded so that `No Liver Disease` is the reference level and the model estimates the probability of `Liver Disease`.

### Coefficients

| Predictor   | Estimate  | Std. Error | z value | Pr(>\|z\|) | Sig. |
|-------------|-----------|------------|---------|-----------|------|
| (Intercept) | -0.6212   | 0.6554     | -0.948  | 0.343     |      |
| TB          | 0.0138    | 0.1041     | 0.132   | 0.895     |      |
| DB          | 0.5524    | 0.2788     | 1.981   | 0.048     | *    |
| Alkphos     | 0.0011    | 0.0008     | 1.458   | 0.145     |      |
| Sgpt        | 0.0094    | 0.0049     | 1.930   | 0.054     | .    |
| Sgot        | 0.0029    | 0.0032     | 0.932   | 0.351     |      |
| TP          | 0.3641    | 0.1773     | 2.053   | 0.040     | *    |
| ALB         | -0.6632   | 0.2557     | -2.594  | 0.009     | **   |

Significance codes: `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1. Coefficients are on the log-odds scale; exponentiate for odds ratios (see below). The Fisher scoring algorithm converged in 7 iterations.

### Model significance

- Null deviance: 684.11 → Residual deviance: 569.60
- Likelihood-ratio test: chi-square ≈ 114.5 (df = 7), p < 0.001 → the model is significantly better than the null.
- McFadden pseudo-R² ≈ 0.17 (modest, typical for routine blood-panel data).
- AIC: 585.6

### Significant predictors (odds ratios, 95% CI)

- **ALB (albumin): OR ≈ 0.52** (95% CI 0.31–0.85, p = 0.009) — strongest effect, protective. Higher albumin roughly halves the odds of disease, consistent with impaired liver synthetic function at low albumin.
- **TP (total protein): OR ≈ 1.44** (95% CI 1.02–2.04, p = 0.040) — risk factor.
- **DB (direct bilirubin): OR ≈ 1.74** (p = 0.048) — risk factor, but borderline: its profile-likelihood CI (0.84–2.89) just includes 1, so the effect is suggestive rather than firmly established.

### Borderline / non-significant predictors

- **Sgpt / ALT:** OR ≈ 1.01 per unit, p = 0.054 — borderline, just above the 0.05 threshold after deduplication (small per-unit effect due to the wide enzyme scale).
- **TB (total bilirubin):** p = 0.90 — redundant given DB (the two bilirubin measures are strongly related; only direct bilirubin carries unique signal).
- **Sgot / AST:** p = 0.35.
- **Alkphos (alkaline phosphatase):** p = 0.15.

### Diagnostics

- **Multicollinearity:** no problematic multicollinearity — all VIF < 4 (max 3.90 for ALB, 3.76 for TP), tolerance > 0.25, so coefficient estimates and their significance are stable.
- **Linearity of the logit:** checked by adding log-transformed versions of each continuous predictor and testing whether they improve the model. The block of log terms was jointly non-significant — likelihood-ratio test (deviance 569.60 → 565.73, df = 7) gives chi-square ≈ 3.87, p ≈ 0.79, and AIC worsens (585.6 → 595.73). Non-significant log terms indicate the linearity-of-the-logit assumption holds, so the simpler model without transformations is preferred.

### Interpretation and limitations

The profile of **low albumin combined with elevated direct bilirubin** is most associated with liver disease, matching clinical expectations. Caveats: the model is fit on all 570 rows without a train/test split, so it measures associations rather than validated predictive accuracy; the classes are imbalanced (~71% positive), and a few effects (DB, Sgpt) are borderline. Next steps: fit a reduced model (dropping `TB`, `Sgot`, `Alkphos`) and compare via AIC / `anova(test="Chisq")`.

## Repository Structure

```
Liver_Disease_Prediction/
├── Liver_Disease_Prediction.ipynb   # Full analysis notebook
├── liver_patient_dataset.csv        # Dataset
└── README.md                        # This file
```

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Liver_Disease_Prediction.ipynb
```

Point the data loader at the local `liver_patient_dataset.csv`, then run all cells. (The notebook's original Kaggle download URL is a temporary signed link and may have expired.)

## Skills Demonstrated

- Leakage-aware machine learning: splitting before scaling and feature selection.
- Handling class imbalance with balanced class weights and appropriate metrics (balanced accuracy, ROC-AUC) rather than raw accuracy.
- Cross-validation-based threshold tuning instead of tuning on the test set.
- Multicollinearity-driven feature selection.
- Interpretable modeling via odds ratios.
- Honest, critical evaluation of model strengths and limitations.
