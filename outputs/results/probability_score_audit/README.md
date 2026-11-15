# Held-out GP predictive scores

Two objectives, two scenes, five training subsets and a fixed 100-point holdout.
NLL and CRPS use the mean and population standard deviation of each training
subset. A constant Gaussian fitted to those responses is the reference model.

`per_subsample.csv` and `metrics.json` contain the GP scores and diagnostics.
`gaussian_reference_per_subsample.csv` and `gaussian_reference_summary.csv`
contain reference scores and CRPS reductions for all scene/objective pairs.
Relative reductions in the summary use ratios of the mean scores.

Reproduce with `python code/experiments/analyze_predictive_scores.py`.
