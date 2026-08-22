## Adsorbate Examples

- `configs/`: YAML inputs for CMC and GCMC workflows
- `runners/`: ad hoc legacy examples
- `analysis/`: post-processing helpers
- `data/`: small example structures

Typical commands:

```bash
gcmc-run-adsorbate-cmc --config examples/adsorbate/configs/adsorbate_cmc.yaml
gcmc-run-adsorbate-gcmc --config examples/adsorbate/configs/adsorbate_gcmc.yaml
gcmc-run-adsorbate-gcmc-scan --config examples/adsorbate/configs/adsorbate_gcmc_scan.yaml
gcmc-analyze-adsorption-motifs --traj <adsorbate-run-directory> --out-dir motif_analysis
python3 examples/adsorbate/analysis/analyze_adsorbate_gcmc_scan.py --scan-dir <scan_output_dir>
```
