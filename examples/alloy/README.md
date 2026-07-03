## Alloy Examples

- `configs/`: YAML inputs for alloy CMC and replica-exchange workflows
- `runners/`: Slurm examples
- `analysis/`: ordering, motif, adsorption-site, and SRO analysis scripts
- `data/`: small example structures

Typical commands:

```bash
gcmc-run-alloy-cmc --config examples/alloy/configs/alloy_cmc.yaml
gcmc-run-alloy-pt --config examples/alloy/configs/alloy_pt.yaml
gcmc-run-alloy-pt --config examples/alloy/configs/alloy_pt_ray.yaml
```

Live trajectory convergence check:

```bash
PYTHONPATH=. python3 examples/alloy/analysis/check_mxene_convergence.py \
  --config config.yaml \
  --traj "replica_*K.traj" \
  --start 0 \
  --step 10 \
  --n-blocks 5 \
  --out-dir convergence_check
```

The convergence check writes block CSV files, `convergence_report.md`, and
quick-look figures under `convergence_check/plots/`.  Use `--no-plots` to skip
plot generation on minimal environments without matplotlib.
