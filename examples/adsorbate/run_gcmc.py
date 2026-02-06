from ase.io import read
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from gcmc import GCMC, analyze_and_plot

# Read MXene structure 
atoms = read('./POSCAR')
z_cut = 13.0  # fix bottom layers
fixed_indices = [atom.index for atom in atoms if atom.position[2] < z_cut]
atoms.set_constraint(FixAtoms(indices=fixed_indices))

# Setup calculator
mace_models = './ft-omat_0-00_stagetwo.model'
mace_calc = MACECalculator(model_paths=[mace_models], device='cuda')

# Run GCMC
print("Starting GCMC...")
gcmc = GCMC(
    atoms=atoms,
    calculator=mace_calc,
    mu=-0.4,
    T=300,
    element='Cu',
    nsteps=50,
    relax=False,
    relax_steps=100,
    T_anneal=None,
    nsteps_anneal=10,
    max_layers=5
)
gcmc.run()


# Analyze clusters
#analyze_and_plot('gcmc_full.traj', cutoff=3.0, element='Cu')
