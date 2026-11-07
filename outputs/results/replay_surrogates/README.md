# FLAT final-parameter replay

Six scenes; GP, RF, MLP; four matched original optimization runs (101, 202, 303, 404); ten shared SUMO seeds (10001 through 10010).

720 completed replays, no failures. 144 compatible earlier rows reused; 576 new replays. Four worker processes; new invocation wall time 1500.13 seconds.

Parameters are fixed throughout. Every parameter source has a complete ten-dimensional vector and an error_at_budget value matching comparison_summary_main.csv. The four optimization runs were selected by common parameter availability, before replay. This is a subset of the original five-run experiment.

## Files

- manifest.json: complete parameter vectors, source/input hashes, protocol and software provenance.
- per_replay.csv: all 720 individual outcomes, including J_b, time-profile speed RMSE and speed W1; duplicate per-row JSON files are omitted.
- per_optimization_seed.csv: average ten simulation seeds within each fixed parameter set.
- summary.csv: per-scene means and sample standard deviations across four original optimization runs.
- original_vs_replay.csv: matched original search observations and replay means.
- scene_comparison.csv and global_comparison.csv: scene-level and equal-scene aggregate comparisons.
- analysis.json: numerical summary and conditional Monte Carlo precision.
- run_accounting.json: execution cost and failure list.

## Interpretation

Six-scene replay mean J_b: GP 0.256224, RF 0.266131, MLP 0.266809. Maximum relative backend gap: 4.131%. All three search-to-replay mean shifts are approximately +0.026. GP has the lowest aggregate mean in both the first-five and last-five seed blocks. RF and MLP exchange aggregate order between those blocks.

These replays evaluate new simulation realizations at fixed parameters. They are not held-out traffic data, nor the appendix's held-out parameter-point GP prediction test. Simulation repetitions are averaged within parameter sets; they do not create additional optimization repetitions or traffic scenes.

## Scope

Only the three FLAT backends with four common complete parameter records are
included. Original external-baseline parameters were unavailable or belonged to
a different batch; no replacement optimization was run. See the frozen manifest
for every parameter vector and source hash. The public replay entry point loads
these vectors directly and validates the recorded simulation inputs.

## Commands

python code/experiments/run_replay_surrogates.py --preflight
python code/experiments/run_replay_surrogates.py --workers 4 --resume
python code/experiments/analyze_replay_surrogates.py
