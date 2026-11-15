# Reference inverse-variance weights

Compute per-vehicle reference weights from the installed SIND and UTE datasets:

```sh
python code/calibration/compute_ivw_weights.py
python code/calibration/test_compute_ivw_weights.py
```

Use `--scenes Tianjin XAM-N6` to select scenes and `--output PATH` for the JSON
output. Existing output files are not overwritten.

SIND records are grouped by `track_id`, UTE records by `vehicleID`. The script
uses the complete observed file and the existing eight-component
`compute_features` definition, including speed units and longitudinal
acceleration conversion. Each vehicle contributes one fingerprint. Tracks with
fewer than 20 samples or nonfinite measurements are excluded and counted.

For each component, sample variance across vehicles (ddof=1) is inverted and
normalized across components to sum to one. Variances use raw feature units
(m/s, m/s², and stop fraction), so the weighting depends on units and can place
large weight on nearly constant components. Zero variance raises an error;
`--variance-floor` supplies an explicit lower bound if needed.

Output includes feature order, input hashes, vehicle counts, variances, weights,
and differences from the configured weights. It contains no vehicle records.
The script does not change calibration defaults. To evaluate a new weight
vector, pass it to `feature_error(real_features, simulated_features, weights)`.
