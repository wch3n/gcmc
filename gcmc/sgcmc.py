import numpy as np
import logging
import os
import pickle
from typing import Dict, Any, List, Optional
from ase import Atoms
from ase.io import Trajectory, write, read
from scipy.spatial import cKDTree
from .alloy_cmc import AlloyCMC

logger = logging.getLogger("mc")


class SemiGrandAlloyMC(AlloyCMC):
    """
    Semi-Grand Canonical Monte Carlo for Alloys (Fixed N, Variable Composition).

    Performs transmutation moves (A -> B) controlled by chemical potential differences.
    Calculates Thermodynamic Susceptibility (composition fluctuations) automatically.

    Args:
        atoms: Initial ASE Atoms object.
        calculator: ASE calculator.
        chem_pots: Dictionary of chemical potentials (eV), e.g., {'Ti': 0.0, 'Mo': 0.5}.
        T: Temperature (K).
        swap_elements: List of elements allowed to transmute.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        chem_pots: Dict[str, float],
        T: float = 300,
        **kwargs,
    ):
        super().__init__(atoms, calculator, T=T, **kwargs)
        self.chem_pots = chem_pots

        # Validate chemical potentials.
        for el in self.swap_elements:
            if el not in self.chem_pots:
                raise ValueError(f"Missing chemical potential for element '{el}'")

        # Initialize energies.
        self.raw_potential_energy = self.atoms.get_potential_energy()
        self.e_old = self._calculate_grand_potential()

    def _calculate_grand_potential(self) -> float:
        """Calculate Phi = E - sum(mu_i * N_i)."""
        mu_sum = 0.0
        symbols = self.atoms.get_chemical_symbols()
        for s in symbols:
            if s in self.chem_pots:
                mu_sum += self.chem_pots[s]
        return self.raw_potential_energy - mu_sum

    def run(
        self,
        nsweeps: int,
        traj_file: str,
        interval: int = 10,
        sample_interval: int = 1,
        equilibration: int = 0,
    ) -> Dict[str, Any]:

        # Trajectory setup.
        self.traj_file = traj_file

        if os.path.exists(self.traj_file) and os.path.getsize(self.traj_file) > 0:
            mode = "a"
        else:
            mode = "w"
        traj_writer = Trajectory(self.traj_file, mode)

        # Initialize statistics.
        # Only reset when starting fresh (sweep 0).
        if self.sweep == 0:
            self.sum_phi = 0.0
            self.sum_phi_sq = 0.0
            self.sum_E = 0.0
            self.sum_E_sq = 0.0
            self.n_samples = 0
            self.accepted_moves = 0
            self.total_moves = 0
            self.sum_N = {el: 0.0 for el in self.swap_elements}
            self.sum_N_sq = {el: 0.0 for el in self.swap_elements}

        # Pre-calculate current counts.
        current_symbols = self.atoms.get_chemical_symbols()
        current_counts = {el: current_symbols.count(el) for el in self.swap_elements}
        moves_per_sweep = len(self.swap_indices)

        # Main loop.
        start_sweep = self.sweep
        end_sweep = start_sweep + nsweeps

        for sweep in range(start_sweep, end_sweep):
            for i in range(moves_per_sweep):
                self.total_moves += 1

                # 1. Propose move.
                idx = self.rng.choice(self.swap_indices)
                old_sym = self.atoms.symbols[idx]

                candidates = [el for el in self.swap_elements if el != old_sym]
                if not candidates:
                    continue
                new_sym = self.rng.choice(candidates)

                self.atoms.symbols[idx] = new_sym

                # 2. Relax and calculate energy.
                if self.relax:
                    atoms_trial, conv = self.relax_structure(
                        self.atoms.copy(), move_ind=[self.sweep, i]
                    )
                    e_new_raw = self.get_potential_energy(atoms_trial) if conv else 1e9
                else:
                    e_new_raw = self.get_potential_energy(self.atoms)

                delta_mu = self.chem_pots[new_sym] - self.chem_pots[old_sym]
                delta_E = e_new_raw - self.raw_potential_energy
                delta_phi = delta_E - delta_mu

                # 3. Acceptance.
                if delta_phi < 0 or (
                    self.rng.random() < np.exp(-delta_phi / (8.617e-5 * self.T))
                ):
                    # Accept move.
                    self.raw_potential_energy = e_new_raw
                    self.e_old += delta_phi
                    current_counts[old_sym] -= 1
                    current_counts[new_sym] += 1
                    self.accepted_moves += 1

                    if self.relax:
                        self.atoms.positions = atoms_trial.positions
                        self.atoms.cell = atoms_trial.cell  # Update cell.
                        if self.swap_mode in ["neighbor", "hybrid"]:
                            self.tree = cKDTree(self.atoms.get_positions())
                else:
                    # Reject move.
                    self.atoms.symbols[idx] = old_sym

            self.sweep += 1

            # Sampling.
            if sweep >= equilibration and (sweep + 1) % sample_interval == 0:
                self.n_samples += 1
                self.sum_phi += self.e_old
                self.sum_phi_sq += self.e_old**2
                self.sum_E += self.raw_potential_energy
                self.sum_E_sq += self.raw_potential_energy**2

                for el in self.swap_elements:
                    cnt = current_counts[el]
                    self.sum_N[el] += cnt
                    self.sum_N_sq[el] += cnt**2

            # Output and checkpointing.
            if (sweep + 1) % interval == 0:
                traj_writer.write(self.atoms)

                with open(self.thermo_file, "a") as f:
                    f.write(f"{self.sweep} {self.e_old:.6f}\n")

                # Save checkpoint.
                if (
                    self.checkpoint_interval > 0
                    and self.sweep % self.checkpoint_interval == 0
                ):
                    self._save_checkpoint()

                # Console logging with concentration.
                acc = (
                    (self.accepted_moves / self.total_moves * 100)
                    if self.total_moves
                    else 0.0
                )
                if self.n_samples > 0:
                    avg_phi = self.sum_phi / self.n_samples
                else:
                    avg_phi = self.e_old

                # Format concentration string (e.g., "Ti: 0.50 | Mo: 0.50").
                total_sites = moves_per_sweep
                conc_str = " | ".join(
                    [
                        f"{el}: {current_counts[el]/total_sites:.2f}"
                        for el in self.swap_elements
                    ]
                )

                logger.info(
                    f"T={self.T:4.0f}K | {self.sweep:6d} | "
                    f"Phi: {self.e_old:10.4f} | Avg: {avg_phi:10.4f} | "
                    f"Acc: {acc:4.1f}% | {conc_str}"
                )

        traj_writer.close()

        # Final results.
        results = self._compute_final_stats(current_counts)
        return results

    def _compute_final_stats(self, current_counts):
        """Helper to package results dictionary."""
        results = {
            "T": self.T,
            "acceptance": (
                (self.accepted_moves / self.total_moves * 100)
                if self.total_moves
                else 0.0
            ),
        }
        if self.n_samples > 0:
            kB = 8.617333e-5
            avg_phi = self.sum_phi / self.n_samples
            avg_E = self.sum_E / self.n_samples
            var_phi = (self.sum_phi_sq / self.n_samples) - (avg_phi**2)
            cv = var_phi / (kB * self.T**2)

            susceptibility = {}
            composition_avg = {}
            for el in self.swap_elements:
                avg_N = self.sum_N[el] / self.n_samples
                var_N = (self.sum_N_sq[el] / self.n_samples) - (avg_N**2)
                composition_avg[el] = avg_N
                susceptibility[el] = var_N / (kB * self.T)

            results.update(
                {
                    "energy": avg_phi,
                    "potential_energy": avg_E,
                    "cv": cv,
                    "composition": composition_avg,
                    "susceptibility": susceptibility,
                }
            )
        else:
            results.update(
                {
                    "energy": self.e_old,
                    "potential_energy": self.raw_potential_energy,
                    "cv": 0.0,
                    "composition": current_counts,
                    "susceptibility": {el: 0.0 for el in self.swap_elements},
                }
            )
        return results

    # Custom checkpoint methods.
    def _save_checkpoint(self):
        """
        Saves SGCMC-specific state (Composition/Grand Potential stats)
        in addition to standard MC state.
        """
        # 1. Save atoms.
        chk_traj = self.traj_file.replace(".traj", "_checkpoint.traj")
        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        write(chk_traj, atoms_copy)

        # 2. Save full state.
        state = {
            "sweep": self.sweep,
            "e_old": self.e_old,
            "raw_potential_energy": self.raw_potential_energy,
            "rng_state": self.rng.bit_generator.state,
            "T": self.T,
            # General stats.
            "sum_E": self.sum_E,
            "sum_E_sq": self.sum_E_sq,
            "n_samples": self.n_samples,
            "total_moves": self.total_moves,
            "accepted_moves": self.accepted_moves,
            # SGCMC-specific stats.
            "sum_phi": self.sum_phi,
            "sum_phi_sq": self.sum_phi_sq,
            "sum_N": self.sum_N,
            "sum_N_sq": self.sum_N_sq,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"SGCMC Checkpoint saved at sweep {self.sweep}")

    def _load_checkpoint(self):
        """
        Restores SGCMC state.
        """
        if not os.path.exists(self.checkpoint_file):
            return

        # 1. Restore atoms.
        chk_traj = self.traj_file.replace(".traj", "_checkpoint.traj")
        if os.path.exists(chk_traj):
            self.atoms = read(chk_traj)
        self.atoms.calc = self.calculator

        # 2. Restore state.
        with open(self.checkpoint_file, "rb") as f:
            state = pickle.load(f)

        self.sweep = state.get("sweep", 0)
        self.e_old = state.get("e_old", None)
        self.raw_potential_energy = state.get("raw_potential_energy")
        self.T = state.get("T", self.T)
        self.rng.bit_generator.state = state["rng_state"]

        self.total_moves = state.get("total_moves", 0)
        self.accepted_moves = state.get("accepted_moves", 0)

        self.sum_E = state.get("sum_E", 0.0)
        self.sum_E_sq = state.get("sum_E_sq", 0.0)
        self.n_samples = state.get("n_samples", 0)

        if "sum_phi" in state:
            self.sum_phi = state["sum_phi"]
            self.sum_phi_sq = state["sum_phi_sq"]
            self.sum_N = state["sum_N"]
            self.sum_N_sq = state["sum_N_sq"]

        logger.info(f"SGCMC Resumed from sweep {self.sweep}")
