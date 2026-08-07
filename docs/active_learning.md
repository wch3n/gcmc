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

Prepared adsorbate-CMC campaigns can be supplied directly instead of listing
every nested trajectory. For example:

```bash
gcmc-active-learning-select \
  --input /path/to/oer/lg/oh \
  --temperature 300 \
  --temperature 539 \
  --temperature 800 \
  --trajectory-kind sampled \
  --trajectory-kind accepted \
  --trajectory-kind rejected \
  --min-per-trajectory-kind 2 \
  --min-per-temperature 1 \
  --input-tail-fraction 0.5 \
  --max-frames-per-trajectory 50 \
  --frame-stride 2 \
  --train-size 8 \
  --test-size 4 \
  --model-path /path/to/committee \
  --calculator symmetrix \
  --device cuda \
  --output-root ./pbe0-rvv10_8
```

Directory inputs are searched recursively for the sampled
`replica_<temperature>K[_{accepted,rejected,attempted}].traj` files, so all
available `runs/snapshot_*/seed_*/results` branches are merged automatically.
The default pools are `sampled`, `accepted`, and `rejected`. They may be
selected explicitly by repeating `--trajectory-kind`; `attempted` is opt-in
because it can contain unphysical proposals rejected before an energy call.
An explicitly requested kind must exist, whereas unavailable kinds from the
implicit default produce a warning and are skipped.
Rejected geometry-filter failures without finite `mc_energy_eV` metadata are
also excluded by default. `--include-unevaluated-rejected` overrides this for
special diagnostics. `--min-per-trajectory-kind` reserves a minimum number of
selected structures for every available requested pool before SOAP max-min
selection fills the remaining slots. This prevents a high-uncertainty rejected
pool from completely displacing sampled or accepted structures.
`--min-per-temperature` provides the analogous safeguard across the selected
replica temperatures.

`--temperature` may be repeated; omit it to include every replica
temperature. The tail fraction discards the early part of each trajectory,
while the per-trajectory cap samples the remaining indices evenly and prevents
longer or more advanced runs from dominating the committee evaluation.
Snapshot, seed, replica temperature, trajectory kind, trajectory, and original
frame index are retained in `selection.csv` and the selected ASE trajectories.
Accepted and rejected pools are available only when the CMC run has
`write_debug_trajs: true` (or the corresponding individual trajectory
switches) enabled.

For MLIP validation, the individual switches avoid the much larger attempted
pool:

```yaml
cmc:
  write_debug_trajs: false
  write_attempted_traj: false
  write_accepted_traj: true
  write_rejected_traj: true
  debug_traj_interval: 10
```

The interval is applied to proposal events before they are dispatched to the
accepted or rejected file. Increase it if trajectory storage becomes
significant.

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

With `--recursive`, evaluation is deliberately restricted to numbered
directories matching `al_NNN/test/NNN`. The corresponding `train` directories
and unrelated numbered directories are excluded, so the resulting history is
based only on structures that remain held out from fine-tuning. Without
`--recursive`, `--root` is treated as one flat directory containing the
requested numbered calculations.

The command writes per-configuration committee metrics, aggregate summaries,
a learning curve, an ASE trajectory containing committee metadata, and
timestamped history snapshots. It also writes two figures in both PDF and PNG
formats:

- `mace_committee_diagnostics`: relative-energy and force parity plots plus
  energy- and force-uncertainty calibration plots
- `mace_learning_curve`: round-specific and cumulative energy/force errors

Relative energies are centered separately within each chemical formula. This
removes composition-dependent constant energy offsets and emphasizes the
energy differences that control fixed-composition MC sampling. Absolute energy
errors remain available in the CSV and JSON outputs. Use `--no-plots` to skip
figure generation, or `--plot-output` and `--learning-curve-plot` to change the
output prefixes. Generated figures are copied into the timestamped history
snapshot.

A single model is accepted for ordinary error evaluation, although
committee-disagreement metrics only become meaningful with multiple models.

Use `--calculator mace` or `--calculator symmetrix` when a model directory
contains both formats. With `--calculator auto`, mixed formats are rejected to
avoid accidentally evaluating two representations of the same model.
