from ase.io import read
from mace.calculators import MACECalculator
from gcmc.alloy_cmc import AlloyCMC
from gcmc.utils import initialize_alloy_sublattice

# Set up initial system.
pristine = read("POSCAR.Ti2CO2")
atoms = initialize_alloy_sublattice(
    pristine,
    "Ti",
    {"Ti": 0.4, "Mo": 0.3, "Zr": 0.3},
)

# Set up calculator.
calc = MACECalculator(model_paths=["mxene.model"], device="cuda")
neighbor_backend = "auto"  # Use matscipy if installed, otherwise ASE.

# Baseline CMC (swap moves only).
mc = AlloyCMC(
    atoms=atoms.copy(),
    calculator=calc,
    T=600,
    swap_elements=["Ti", "Mo", "Zr"],
    swap_mode="neighbor",
    neighbor_backend=neighbor_backend,
    relax=True,
    traj_file="cmc_only.traj",
    thermo_file="cmc_only.dat",
)
stats = mc.run(nsweeps=200, traj_file="cmc_only.traj", interval=10, equilibration=20)
print("CMC only:", stats)

# Hybrid MC/MD with NVE MD proposals.
mc_hybrid_nve = AlloyCMC(
    atoms=atoms.copy(),
    calculator=calc,
    T=600,
    swap_elements=["Ti", "Mo", "Zr"],
    swap_mode="neighbor",
    neighbor_backend=neighbor_backend,
    relax=True,
    enable_hybrid_md=True,
    md_move_prob=0.2,  # 20% MD proposals and 80% swap proposals.
    md_ensemble="nve",
    md_steps=50,
    md_timestep_fs=1.0,
    traj_file="hybrid_nve.traj",
    thermo_file="hybrid_nve.dat",
)
stats_nve = mc_hybrid_nve.run(
    nsweeps=200, traj_file="hybrid_nve.traj", interval=10, equilibration=20
)
print("Hybrid NVE:", stats_nve)

# Hybrid MC/MD with Langevin MD proposals.
mc_hybrid_langevin = AlloyCMC(
    atoms=atoms.copy(),
    calculator=calc,
    T=600,
    swap_elements=["Ti", "Mo", "Zr"],
    swap_mode="neighbor",
    neighbor_backend=neighbor_backend,
    relax=True,
    enable_hybrid_md=True,
    md_move_prob=0.2,
    md_ensemble="langevin",
    md_steps=50,
    md_timestep_fs=1.0,
    md_friction=0.01,
    traj_file="hybrid_langevin.traj",
    thermo_file="hybrid_langevin.dat",
)
stats_langevin = mc_hybrid_langevin.run(
    nsweeps=200, traj_file="hybrid_langevin.traj", interval=10, equilibration=20
)
print("Hybrid Langevin:", stats_langevin)
