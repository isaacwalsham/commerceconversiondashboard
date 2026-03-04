# Executive Summary

## Headline
The selected model (`random_forest`) ranks likely converters well and gives a useful threshold for targeting.

## Key Metrics
- PR-AUC: 0.724
- ROC-AUC: 0.926
- Brier score: 0.0847
- Best value threshold: 0.09
- Max expected value (relative units): 21,775.5

## Prioritization Signal
- Top decile conversion rate: 74.49%
- Top decile lift vs baseline: 4.81x
- Cumulative conversions captured by top decile: 48.17%

## Recommended Actions
1. Use `0.09` as a starting threshold for "likely to convert".
2. Use decile rank to focus attention on top-scoring sessions first.
3. Track calibration monthly; recalibrate or retrain if gap widens.

## Caveats
- This is a predictive model, not a causal model.
- Expected-value results depend on the conversion value/contact cost assumptions.
