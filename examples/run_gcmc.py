from ase.io import read
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from gcmc import MultiLayerSiteGCMC, find_fcc_hollow_sites, analyze_and_plot

mxene_poscar = './POSCAR'
mace_models = './ft-omat_0-00_stagetwo.model'
chemical_potential = -3.0 # bulk ref: -3.0
temperature = 300
nsteps = 1000
traj_file = f'cu_gcmc_{temperature}_{chemical_potential}.traj'
relax = True
relax_steps = 200
n_layers = 5
layer_spacing = 2.2
fcc_offset = 1.8
cu_symbol = 'Cu'
cutoff = 3.0

# --- 1. Read MXene structure ---
slab = read(mxene_poscar)
z_cut = 13.0  # set to your desired value
fixed_indices = [atom.index for atom in slab if atom.position[2] < z_cut]
slab.set_constraint(FixAtoms(indices=fixed_indices))

# --- 2. Find FCC hollow sites ---
fcc_sites = find_fcc_hollow_sites(slab, fcc_offset=fcc_offset)
print(f"Found {len(fcc_sites)} FCC hollow sites.")

# --- 3. Build multilayer adsorption sites ---
all_sites = []
for site in fcc_sites:
    for l in range(n_layers):
        all_sites.append([site[0], site[1], site[2] + l * layer_spacing])

# --- 4. Setup calculator ---
calc = MACECalculator(model_paths=[mace_models], device='cuda')

# --- 5. Run GCMC ---
print("Starting GCMC...")
gcmc = MultiLayerSiteGCMC(
    atoms=slab,
    calculator=calc,
    mu=chemical_potential,
    T=temperature,
    all_sites=all_sites,
    element=cu_symbol,
    nsteps=nsteps,
    relax=relax,
    relax_steps=relax_steps,
    traj_file=traj_file
)
gcmc.run()

# --- 6. Analyze clusters ---
analyze_and_plot(traj_file, cutoff=cutoff, element=cu_symbol)
