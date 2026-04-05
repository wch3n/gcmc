## Adsorbate Examples

- `configs/`: YAML inputs for CMC and GCMC workflows
- `runners/`: thin Python launchers around the YAML workflows
- `analysis/`: post-processing helpers
- `data/`: small example structures

Typical commands:

```bash
python3 examples/adsorbate/runners/run_adsorbate_cmc.py --config examples/adsorbate/configs/adsorbate_cmc.yaml
python3 examples/adsorbate/runners/run_adsorbate_gcmc.py --config examples/adsorbate/configs/adsorbate_gcmc.yaml
python3 examples/adsorbate/runners/run_adsorbate_gcmc_scan.py --config examples/adsorbate/configs/adsorbate_gcmc_scan.yaml
python3 examples/adsorbate/analysis/analyze_adsorbate_gcmc_scan.py --scan-dir <scan_output_dir>
```
