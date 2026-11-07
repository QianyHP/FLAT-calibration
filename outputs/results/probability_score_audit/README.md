# Appendix A probability-score audit

No SUMO runs, new optimization, GP recalibration, or changes to parameter bounds/IVW/Lipschitz analysis.

The original 400-point landscapes, fixed 100-point holdout, N=100 training samples, five training subsets, and original GP implementation are reused. Raw NLL and CRPS means reproduce the original Appendix A table to better than 1e-8.

For each fitted GP, the training-response population mean m and standard deviation s define standardized held-out observations (y-m)/s, predictive means (mu-m)/s, and standard deviations sigma/s. Holdout moments are not used. The exact identities NLL_standardized = NLL_raw - log(s) and CRPS_standardized = CRPS_raw/s are verified for every fit.

- metrics.json: raw and standardized scores, MACE, coverage, and source hashes.
- per_subsample.csv: all 20 fit-level results and training-response moments.
- predictions.csv: all 2,000 held-out predictive means, SDs, and observed outcomes in original units.
- splits.json: exact training and held-out indices.

NLL and CRPS assess predictive distribution quality, not interval calibration alone. The interval diagnostics are mixed across the two scenes; they do not support the former statement that only J_b has a well-calibrated posterior. The paper retains the two existing score rows, standardizes their scale, and describes probabilistic fit instead.

The original landscape_calibration_v2.json remains unchanged. Run python code/experiments/audit_probability_scores.py to reproduce this audit.
