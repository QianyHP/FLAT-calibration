# FLAT final-parameter replay

Six scenes; GP, RF, MLP; four matched original optimization runs (101, 202, 303, 404); ten shared SUMO seeds (10001 through 10010).

720 completed replays, no failures. 144 compatible earlier rows reused; 576 new replays. Four worker processes; new invocation wall time 1500.13 seconds.

Parameters are fixed throughout. Every parameter source has a complete ten-dimensional vector and an error_at_budget value matching comparison_summary_main.csv. The four optimization runs were selected by common parameter availability, before replay. This is a subset of the original five-run experiment.

## Files

- manifest.json: all 72 frozen parameter vectors, source/input hashes, and protocol.
- per_replay.csv: all 720 individual outcomes, including J_b and auxiliary diagnostics.
- summary.csv: per-scene means and SDs over the four optimization repeats.
- global_comparison.csv: equal-scene backend means and simulation-seed checks.

The analysis script reconstructs per-optimization, original-versus-replay,
scene-comparison, and numerical analysis files from these records. Those derived
files and duplicate per-row JSON records are omitted from the published tree.
The original computation used 144 compatible earlier rows and 576 new replays;
four workers completed the new invocation in 1500.13 seconds, with no failures.

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
