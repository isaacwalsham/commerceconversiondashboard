# Executive Summary: Conversion Intelligence

## Headline
The selected model (`random_forest`) identifies high-intent sessions with strong ranking quality and supports commercial threshold optimization.

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
1. Route sessions above `0.09` to high-intent interventions.
2. Use decile rank to allocate retargeting budget by value density.
3. Track calibration monthly; recalibrate or retrain if gap widens.

## Caveats
- Use this model for prioritization support, not as a standalone decision engine.
- Validate expected-value assumptions (conversion value/contact cost) for your business context.
