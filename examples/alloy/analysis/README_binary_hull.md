# Binary Alloy Hull Check

Use `check_binary_alloy_hull.py` to compare sampled binary-alloy CMC/PT
energies against endpoint tie lines using the same MLIP calculator as the run.

For cluster use, submit the generic Slurm wrapper:

```bash
RUN_DIR=/path/to/binary/alloy/run \
ELEMENT_A=Ti \
ELEMENT_B=Zr \
sbatch /gpfs/home/acad/ucl-modl/wchen/mxene_proj/gcmc/examples/alloy/runners/check_binary_alloy_hull.slurm
```

The wrapper expects:

- `${RUN_DIR}/config.yaml`
- `${RUN_DIR}/results/replica_300K.traj`
- `${RUN_DIR}/results/replica_300K.dat`
- endpoint structures named `POSCAR.${ELEMENT_A}` and `POSCAR.${ELEMENT_B}` in
  the parent system directory, or explicit `ENDPOINT_A` / `ENDPOINT_B` paths.

Useful overrides:

```bash
# Analyze another replica temperature.
TARGET_TEMPERATURE_K=400

# Discard initial samples and thin the trajectory/dat stream.
START=500 STEP=5

# Use explicit endpoint structures.
ENDPOINT_A=/path/to/POSCAR.Ti ENDPOINT_B=/path/to/POSCAR.Mo

# Relax endpoint atomic positions before endpoint energy evaluation.
RELAX_ENDPOINTS=1 ENDPOINT_FMAX=0.03 ENDPOINT_RELAX_STEPS=300

# Recompute sampled alloy frame energies instead of using replica_*.dat.
RECOMPUTE_ALLOY=1 STEP=20

# Quench selected alloy frames to approximate a 0 K hull comparison.
# This implies recomputing alloy energies and can be expensive.
QUENCH_ALLOY=1 RELAX_ENDPOINTS=1 START=500 STEP=20 MAX_FRAMES=50 \
ALLOY_FMAX=0.03 ALLOY_RELAX_STEPS=300

# Print progress every selected alloy frame. Use 10 for less verbose logs.
PROGRESS_INTERVAL=1

# Disable coherent mean-cell endpoint comparison.
COHERENT_ENDPOINTS=0

# Change normalization for non-M2CO2 formula units.
FORMULA_COUNT_ELEMENT=C METALS_PER_FORMULA=2
```

Outputs are written to:

```text
${RUN_DIR}/results/hull_check_${ELEMENT_A}${ELEMENT_B}/
```

Key files:

- `hull_report.md`: short text summary.
- `hull_summary.csv`: min/mean/max hull distances.
- `endpoint_energies.csv`: endpoint energies used for the tie line.
- `alloy_hull_samples.csv`: per-sample hull distances.
- `plots/`: hull-distance plots.

The script reports both `DeltaE` and an ideal configurational correction
`DeltaG_ideal = DeltaE - T S_config` for the sampled composition.

For the 0 K comparison, use `QUENCH_ALLOY=1 RELAX_ENDPOINTS=1`.  The script
performs fixed-cell atomic relaxations of the selected alloy frames and
endpoints, then applies the same ideal configurational free-energy term at the
requested temperature.  At 50:50 composition and 300 K this ideal term is
approximately `-17.9 meV/metal` for a two-metal formula unit.
