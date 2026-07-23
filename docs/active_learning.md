# Active-Learning Tools

The repository distributes two auxiliary atomistic active-learning commands.
They are packaged under `tools.active_learning`, separately from the core
`gcmc` simulation namespace.

Install the optional selection dependencies with:

```bash
pip3 install '.[active-learning]'
```

## Select Configurations

`gcmc-active-learning-select` screens one or more trajectories with a model
committee, ranks committee disagreement and SOAP novelty, selects diverse
training and holdout-test configurations, and prepares static VASP inputs.

```bash
gcmc-active-learning-select \
  --input ../results/replica_301K_rejected.traj \
  --model-path /path/to/committee \
  --calculator symmetrix \
  --output-root .
```

The model path may be a model file or a directory and may be repeated.
Directories are searched non-recursively. Native MACE models use `.model`;
Symmetrix models use `.json`. Selection requires at least two committee
members.

The main selection defaults are:

- trajectory stride: `2`
- training configurations: `8`
- holdout-test configurations: `4`
- uncertainty shortlist: `300`
- SOAP cutoff: `6.0` A
- SOAP basis: `n_max=8`, `l_max=6`
- novelty weight: `0.45`
- hard uncertainty and SOAP-distance thresholds: disabled

Unless `--no-reference-pool` is supplied, the selector discovers
`*_train.xyz` and `*_test.xyz` beside the model directory. Additional pools
can be supplied with `--model-seen-pool` and `--dft-seen-pool`.

## Evaluate a Committee

`gcmc-active-learning-evaluate` compares one or more models against completed
VASP calculations and maintains historical error metrics.

```bash
gcmc-active-learning-evaluate \
  --root . \
  --recursive \
  --model-path /path/to/committee \
  --calculator symmetrix \
  --history-note round_003
```

The command writes per-configuration committee metrics, aggregate summaries,
a learning curve, an ASE trajectory containing committee metadata, and
timestamped history snapshots. A single model is accepted for ordinary error
evaluation, although committee-disagreement metrics only become meaningful
with multiple models.

Use `--calculator mace` or `--calculator symmetrix` when a model directory
contains both formats. With `--calculator auto`, mixed formats are rejected to
avoid accidentally evaluating two representations of the same model.
