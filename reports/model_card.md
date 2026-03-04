# Model Card: Conversion Project

## 1. Model Details
- Model type: `random_forest`
- Objective: predict session-level purchase conversion
- Training timestamp (UTC): 2026-03-04T21:12:29.699808+00:00

## 2. Data
- Dataset: UCI Online Shoppers Purchasing Intention
- Rows: 12,330
- Observed conversion rate: 15.47%

## 3. Preprocessing
- Numeric features: median imputation + standardization
- Categorical features: mode imputation + one-hot encoding
- Split strategy: stratified train/test
- Model selection: 5-fold stratified cross-validation (PR-AUC primary)

## 4. Performance
- ROC-AUC (test, threshold 0.50): 0.926
- PR-AUC (test, threshold 0.50): 0.724
- F1 (test, threshold 0.50): 0.651
- Brier score (test): 0.0847
- Optimized threshold (value-based): 0.09
- F1 (test, optimized threshold): 0.417

## 5. Intended Use
- Rank sessions by purchase likelihood
- Compare threshold trade-offs in one place
- Use as a learning project for classification workflows

## 6. Limitations
- Single public dataset may not represent all verticals/regions
- No causal inference: scores estimate likelihood, not treatment effect
- Data drift (seasonality, channel changes) can reduce performance

## 7. Monitoring Plan
- Weekly: score distribution drift and conversion-rate drift
- Monthly: calibration check and decile lift stability
- Quarterly: retrain and compare metrics against previous runs
