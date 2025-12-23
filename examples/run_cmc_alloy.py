from ase.io import read
from mace.calculators import MACECalculator
from gcmc.alloy_cmc import AlloyCMC
from gcmc.utils import initialize_alloy_sublattice

# setup initial system 
pristine = read('Ti2CO2_pristine.POSCAR')
atoms = initialize_alloy_sublattice(
    pristine, 'Ti', 
    {'Ti': 0.4, 'Mo': 0.3, 'Zr': 0.3}
)

# setup calculator
calc = MACECalculator(model_paths=['mxene.model'], device='cuda')

# initialize MC
mc = AlloyCMC(
    atoms=atoms,
    calculator=calc,
    swap_elements=['Ti', 'Mo', 'Zr'],
    swap_mode='neighbor',      # Use neighbor swaps for better acceptance
    relax=True,
    traj_file='scan_evolution.traj',
    thermo_file='scan_full_history.dat'
)

# Run cooling
# From 1000K to 100K in steps of 50K
mc.run_temperature_scan(
    T_start=1000,
    T_end=100,
    T_step=50,
    sweeps_per_temp=200,     # Run 500 sweeps at each T
    equilibration=10,       # Ignore first 100 sweeps for stats
    scan_file='results.csv', # Saves T vs E vs Cv
    traj_prefix='run'
)
