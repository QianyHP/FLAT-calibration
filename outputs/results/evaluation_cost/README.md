# Serial evaluation cost

Six scenes, one published GP final parameter vector per scene (optimization
run 101), three shared simulation seeds (20001--20003), 18 serial evaluations.
`evaluation_cost.csv` contains individual measurements; the summary reports
scene means, medians and ranges. `protocol.json` records hardware and execution
hashes. The timer includes SUMO startup, TraCI collection, fingerprints and
scoring; observed-reference extraction and surrogate fitting are excluded.

Reproduce with `python code/experiments/measure_evaluation_cost.py --output outputs/results/evaluation_cost_new`.
