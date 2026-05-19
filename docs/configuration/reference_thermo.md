# Reference Thermo Helper

Use the reference thermo helper to relax an isolated molecule, run ASE
vibrations, and emit a CHE config snippet for `IdealGasThermo`.

For OER post-processing, prefer the integrated `reference_thermo:` section in
`gcmc-reaction-postprocess` when `H2`, `H2O`, and optionally `O2` should be
generated with the same calculator before CHE. This standalone runner remains
useful for one-off checks or precomputing a single reference molecule.

## Runner

```bash
gcmc-reference-thermo --config reference_thermo.yaml
```

## Example

```yaml
reference:
  molecule: H2
  vacuum_A: 8.0
  temperature_K: 303.0
  pressure_Pa: 101325.0
  energy_eV: null
  correction_eV: 0.0
  geometry: linear
  symmetrynumber: 2
  spin: 0.0

calculator:
  calculator: symmetrix
  model_file: model.json
  device: cuda
  use_kokkos: true

output:
  output_dir: h2_reference

relaxation:
  enabled: true
  fmax: 0.001
  steps: 1000
  log_file: relax.log
  progress_log: reference_thermo.log
  progress_stdout: true

vibrations:
  delta_A: 0.01
  nfree: 2
  ignore_imag_modes: false
```

For `H2O`, change:

- `reference.molecule: H2O`
- `reference.geometry: nonlinear`

For `O2`, use:

- `reference.molecule: O2`
- `reference.geometry: linear`
- `reference.symmetrynumber: 2`
- `reference.spin: 1.0`

Set `reference.energy_eV` when the electronic reference energy should come
from an external DFT calculation. The helper still relaxes the molecule and
computes the vibrational/ideal-gas terms, but `IdealGasThermo` and the emitted
CHE snippet use the supplied `energy_eV` instead of the calculator energy.
Use `reference.correction_eV` only for an additional molecular offset on top of
the computed gas-phase Gibbs free energy.

## Outputs

Under `output.output_dir` the helper writes:

- `initial.traj`
- `relaxed.traj`
- `summary.yaml`
- `che_snippet.yaml`
- `vibrations/vib.*`

`summary.yaml` records both `reference_energy_eV` (the energy used for thermo)
and `calculated_potential_energy_eV` (the relaxed molecule energy from the
configured calculator). `che_snippet.yaml` can be merged into the `che:` block
of the reaction post-processing config. It contains:

- `<label>_energy_eV`
- `<label>_thermo.enabled: true`
- `<label>_thermo.temperature_K`
- `<label>_thermo.pressure_Pa`
- `<label>_thermo.geometry`
- `<label>_thermo.symmetrynumber`
- `<label>_thermo.spin`
- `<label>_thermo.vib_energies_eV`
- `<label>_correction_eV`, if `reference.correction_eV` is supplied

where `<label>` is `h2`, `h2o`, or `o2`.
