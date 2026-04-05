## Alloy Examples

- `configs/`: YAML inputs for alloy CMC and replica-exchange workflows
- `runners/`: thin Python launchers and the Ray Slurm example
- `analysis/`: ordering, motif, adsorption-site, and SRO analysis scripts
- `data/`: small example structures

Typical commands:

```bash
python3 examples/alloy/runners/run_cmc_alloy.py --config examples/alloy/configs/alloy_cmc.yaml
python3 examples/alloy/runners/run_pt_cmc_alloy.py --config examples/alloy/configs/alloy_pt.yaml
python3 examples/alloy/runners/run_pt_cmc_alloy_ray.py --config examples/alloy/configs/alloy_pt_ray.yaml
```
