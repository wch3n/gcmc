# Adsorption-motif analysis

`gcmc-analyze-adsorption-motifs` classifies tagged adsorbates in sampled CMC/PT
trajectories against the adsorption-site registry of a reference slab.

## Motif definitions

The adsorbate anchor is the requested `--anchor-element`. If a molecular group
contains more than one such atom, the atom closest to the selected surface is
used. The lateral anchor position is matched under periodic boundary conditions
to the nearest reference site.

| Coarse motif | Coordination | Registry detail |
| --- | ---: | --- |
| `atop` | 1-fold | `atop` |
| `bridge` | 2-fold | `bridge` |
| `hollow` | 3-fold | `fcc` or `hcp` |
| `off_site` | 0 | Farther than `--max-site-distance` from every registry site |
| `detached` | 0 | Farther than `--max-anchor-support-distance` from its support atoms |

The detailed tables retain the support atom indices and composition, local
metal-shell composition, neighboring termination count, anchor height, and the
outward displacement of the support atoms relative to the reference slab. A
median surface-layer displacement is removed from the support displacement so a
rigid slab translation is not reported as puckering.

## OH/PT example

Analyze every sampled temperature trajectory below a generated OH run:

```bash
gcmc-analyze-adsorption-motifs \
  --traj runs \
  --site-elements Ti Zr \
  --substrate-elements Ti Zr C \
  --anchor-element O \
  --start 200 \
  --step 2 \
  --out-dir motif_analysis \
  --write-representatives
```

Directory inputs are searched recursively by default. Only sampled files named
`replica_<temperature>K.traj` are included; attempted, accepted, and rejected
debug trajectories are not selected implicitly. For each sampled trajectory,
the adjacent `adsorbate_pt_initial.traj` is used as the reference when present.

Important outputs are:

- `all_families.csv`: atop/bridge/hollow populations aggregated by temperature;
- `all_patterns.csv`: registry subtype and Ti/Zr support-composition populations;
- `all_per_frame.csv`: complete frame assignments and geometric diagnostics;
- `adsorption_motifs_transitions.csv`: motif-to-motif transition counts for each trajectory;
- `adsorption_motifs_representatives.traj`: optional representative full structures.
- `motif_distribution.pdf` and `.png`: family populations versus temperature and
  dominant detailed patterns at the lowest sampled temperature.

Plots are written by default. Use `--no-plot` to suppress them,
`--plot-target-temperature 300` to choose the detailed-pattern panel, and
`--plot-top-patterns 10` to change the number of explicit patterns. When multiple
trajectories are available at one temperature, fractions are averaged with equal
weight per trajectory and the error bars show the standard error over trajectories.

For binary surfaces, the temperature panel retains one population curve per
adsorption family. The point and line-segment colors interpolate from the first
to the second element in `--plot-support-elements Ti Zr`. At each temperature,
the color is the conditional mean support composition among observations of that
family: atop therefore identifies the anchor metal directly, while bridge and
hollow use the composition of their two or three support atoms. Marker and line
style distinguish the adsorption families. Use `--no-plot-color-by-support` to
restore fixed family colors.

The reported populations are trajectory occurrence fractions. If the CMC move
kernel is deliberately biased for configuration discovery, these fractions are
sampling diagnostics rather than rigorous canonical probabilities.
