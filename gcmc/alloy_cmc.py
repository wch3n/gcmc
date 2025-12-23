import logging
import numpy as np
import os
from typing import Any, Optional, List, Union, Dict
from ase import Atoms
from ase.io import read, Trajectory
from scipy.spatial import cKDTree

from .base import BaseMC

logger = logging.getLogger("mc")

class AlloyCMC(BaseMC):
    """
    Canonical Monte Carlo for solid alloys.
    Supports temperature scanning with separate trajectory outputs per step.
    """

    def __init__(
        self,
        atoms: Union[Atoms, str],
        calculator: Any,
        T: float = 300,
        swap_elements: Optional[List[str]] = None,
        swap_mode: str = "global",
        neighbor_cutoff: float = 3.5,
        relax: bool = False,
        relax_steps: int = 10,
        fmax: float = 0.05,
        traj_file: str = "alloy_cmc.traj",
        accepted_traj_file: Optional[str] = "alloy_accepted.traj", # None to disable
        thermo_file: str = "energies.dat",
        checkpoint_interval: int = 100,
        seed: int = 42,
        **kwargs
    ):
        # 1. Handle File Reading
        if isinstance(atoms, str):
            self.atoms = read(atoms)
        else:
            self.atoms = atoms.copy()

        super().__init__(
            atoms=self.atoms,
            calculator=calculator,
            adsorbate_element="X", 
            substrate_elements=(), 
            relax_steps=relax_steps,
            fmax=fmax,
            seed=seed,
            **kwargs
        )

        self.T = T
        self.swap_mode = swap_mode
        self.neighbor_cutoff = neighbor_cutoff
        self.relax = relax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.thermo_file = thermo_file
        self.checkpoint_interval = checkpoint_interval
        
        # Setup Hot Species
        unique_elements = set(self.atoms.get_chemical_symbols())
        if swap_elements is None:
            self.swap_elements = list(unique_elements)
        else:
            self.swap_elements = swap_elements
            
        self.swap_indices = np.array([
            i for i, s in enumerate(self.atoms.get_chemical_symbols())
            if s in self.swap_elements
        ])
        
        if self.swap_mode == "neighbor":
            self.tree = cKDTree(self.atoms.get_positions())

        # Initial State
        self.atoms.calc = self.calculator
        self.e_old = self.atoms.get_potential_energy()
        self.sweep = 0
        self.accepted_moves = 0
        self.total_moves = 0

    def attempt_swap(self) -> Optional[Atoms]:
        """Attempts a swap based on swap_mode."""
        if len(self.swap_indices) < 2:
            return None

        # --- Global Random Swap ---
        if self.swap_mode == "global":
            idx1, idx2 = self.rng.choice(self.swap_indices, size=2, replace=False)

        # --- Nearest Neighbor Swap ---
        elif self.swap_mode == "neighbor":
            idx1 = self.rng.choice(self.swap_indices)
            dists, neighbors = self.tree.query(
                self.atoms.positions[idx1], 
                k=12, 
                distance_upper_bound=self.neighbor_cutoff
            )
            
            valid_neighbors = []
            for n_idx in neighbors:
                if n_idx < len(self.atoms) and n_idx != idx1:
                    if n_idx in self.swap_indices:
                        valid_neighbors.append(n_idx)
            
            if not valid_neighbors:
                return None
            idx2 = self.rng.choice(valid_neighbors)

        # --- Perform Swap ---
        sym1 = self.atoms.symbols[idx1]
        sym2 = self.atoms.symbols[idx2]

        if sym1 == sym2:
            return None

        atoms_new = self.atoms.copy()
        atoms_new.symbols[idx1] = sym2
        atoms_new.symbols[idx2] = sym1
        
        return atoms_new

    def run(
        self, 
        nsweeps: int = 100, 
        log_interval: int = 10, 
        save_interval: int = 1,
        equilibration: int = 0,
        traj_file: Optional[str] = None,          # <--- New override
        accepted_traj_file: Optional[str] = None  # <--- New override
    ) -> Dict[str, float]:
        """
        Run the simulation for a fixed T.
        """
        # Resolve filenames (use method args if provided, else class attributes)
        out_traj = traj_file if traj_file is not None else self.traj_file
        acc_traj = accepted_traj_file if accepted_traj_file is not None else self.accepted_traj_file

        mode = 'a' if os.path.exists(out_traj) else 'w'
        traj = Trajectory(out_traj, mode)
        
        # Only create accepted trajectory if filename is provided
        traj_acc = None
        if acc_traj:
            mode_acc = 'a' if os.path.exists(acc_traj) else 'w'
            traj_acc = Trajectory(acc_traj, mode_acc)
        
        # Initialize Energy File
        if not os.path.exists(self.thermo_file) or os.stat(self.thermo_file).st_size == 0:
            with open(self.thermo_file, "w") as f:
                f.write("# Sweep Energy_eV\n")
        
        moves_per_sweep = len(self.swap_indices)
        
        # Stats accumulators
        sum_E = 0.0
        sum_E_sq = 0.0
        n_samples = 0
        
        logger.info(f"{'Sweep':>6} | {'Energy (eV)':>12} | {'Avg E':>12} | {'Cv (est)':>10} | {'Acc %':>6}")

        for sweep in range(nsweeps):
            # Local sweep 0..N, but we track global persistence
            current_global_sweep = self.sweep + 1
            
            for i in range(moves_per_sweep):
                self.total_moves += 1
                
                # 1. Propose & Relax
                atoms_new = self.attempt_swap()
                if atoms_new is None: continue 

                if self.relax:
                    atoms_new, converged = self.relax_structure(atoms_new, move_ind=[current_global_sweep, i])
                    if not converged: continue

                # 2. Metropolis
                e_new = self.get_potential_energy(atoms_new)
                delta_e = e_new - self.e_old
                
                if delta_e < 0 or (self.rng.random() < np.exp(-delta_e / (8.617e-5 * self.T))):
                    self.atoms = atoms_new
                    self.e_old = e_new
                    self.accepted_moves += 1
                    if traj_acc:
                        traj_acc.write(self.atoms)
            
            # --- End of Sweep ---
            traj.write(self.atoms)
            self.sweep += 1 
            
            # Save Energy
            if current_global_sweep % save_interval == 0:
                with open(self.thermo_file, "a") as f:
                    f.write(f"{current_global_sweep} {self.e_old:.6f}\n")

            # Collect Stats
            if sweep >= equilibration:
                sum_E += self.e_old
                sum_E_sq += (self.e_old ** 2)
                n_samples += 1

            # Logging
            if current_global_sweep % log_interval == 0:
                acc_rate = (self.accepted_moves / self.total_moves * 100) if self.total_moves > 0 else 0.0
                avg_E = sum_E / n_samples if n_samples > 0 else 0.0
                Cv = 0.0
                if n_samples > 1:
                    var_E = (sum_E_sq / n_samples) - (avg_E ** 2)
                    Cv = var_E / (8.617e-5 * self.T**2)
                
                logger.info(
                    f"{current_global_sweep:6d} | {self.e_old:12.4f} | {avg_E:12.4f} | {Cv:10.4f} | {acc_rate:5.1f}"
                )
        
        final_avg_E = sum_E / n_samples if n_samples > 0 else self.e_old
        final_Cv = 0.0
        if n_samples > 1:
            var_E = (sum_E_sq / n_samples) - (final_avg_E ** 2)
            final_Cv = var_E / (8.617e-5 * self.T**2)
        
        # Close trajectories manually to be safe
        traj.close()
        if traj_acc:
            traj_acc.close()

        return {
            "energy": final_avg_E,
            "cv": final_Cv,
            "acceptance": (self.accepted_moves / self.total_moves * 100) if self.total_moves else 0.0
        }

    def run_temperature_scan(
        self,
        T_start: float,
        T_end: float,
        T_step: float,
        sweeps_per_temp: int = 500,
        equilibration: int = 100,
        scan_file: str = "scan_results.dat",
        traj_prefix: str = "traj",
        log_interval=10,
        save_interval=10
    ):
        """
        Perform temperature scan with separate trajectory files per T.
        Accepted trajectories are DISABLED to save space.
        """
        if T_start <= T_end:
            temps = np.arange(T_start, T_end + 1e-9, abs(T_step))
        else:
            temps = np.arange(T_start, T_end - 1e-9, -abs(T_step))

        logger.info(f"Starting Scan: {len(temps)} steps from {T_start}K to {T_end}K")
        
        with open(scan_file, "w") as f:
            f.write("# T    Energy    Cv    Acceptance\n")

        for i, T in enumerate(temps):
            logger.info(f"--- Scan Step {i+1}/{len(temps)}: T = {T:.2f} K ---")
            
            # Update Temperature
            self.T = T
            
            # Define separate trajectory file for this temperature
            step_traj_file = f"{traj_prefix}_{T:.0f}K.traj"
            
            # Run (Disable accepted_traj_file by passing None)
            stats = self.run(
                nsweeps=sweeps_per_temp,
                equilibration=equilibration,
                log_interval=log_interval,
                save_interval=save_interval,
                traj_file=step_traj_file,
                accepted_traj_file=None
            )
            
            with open(scan_file, "a") as f:
                f.write(f"{T:.2f} {stats['energy']:.6f} {stats['cv']:.6f} {stats['acceptance']:.2f}\n")

        logger.info("Temperature scan completed.")
