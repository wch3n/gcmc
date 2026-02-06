import logging
import numpy as np
import os
import pickle
from typing import Any, Optional, List, Union, Dict, Tuple
from ase import Atoms
from ase.io import read, Trajectory
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

from .base import BaseMC

logger = logging.getLogger("mc")


class AlloyCMC(BaseMC):
    """
    Canonical Monte Carlo for solid alloys.
    Optimized for Parallel Tempering.
    """

    def __init__(
        self,
        atoms: Union[Atoms, str],
        calculator: Any,
        T: float = 300,
        swap_elements: Optional[List[str]] = None,
        swap_mode: str = "hybrid",
        neighbor_cutoff: float = 3.5,
        neighbor_backend: str = "ase",
        neighbor_cache: bool = True,
        relax: bool = False,
        relax_steps: int = 10,
        local_relax: bool = False,
        relax_radius: float = 4.0,
        fmax: float = 0.05,
        traj_file: str = "alloy_cmc.traj",
        accepted_traj_file: Optional[str] = "alloy_accepted.traj",
        thermo_file: str = "energies.dat",
        checkpoint_file: str = "restart.pkl",
        checkpoint_interval: int = 100,
        seed: int = 67,
        resume: bool = False,
        enable_hybrid_md: bool = False,
        md_move_prob: float = 0.1,
        md_steps: int = 50,
        md_timestep_fs: float = 1.0,
        md_ensemble: str = "nve",
        md_friction: float = 0.01,
        md_init_momenta: bool = True,
        md_remove_drift: bool = True,
        **kwargs,
    ):
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
            **kwargs,
        )

        self.T = T
        self.swap_mode = swap_mode
        self.neighbor_cutoff = neighbor_cutoff
        self._matscipy_neighbor_list = None
        self.neighbor_backend = self._resolve_neighbor_backend(neighbor_backend)
        self.neighbor_cache = bool(neighbor_cache)
        self.relax = relax
        self.local_relax = local_relax
        self.relax_radius = relax_radius
        self.traj_file = traj_file
        self.thermo_file = thermo_file
        self.checkpoint_file = checkpoint_file
        self.accepted_traj_file = accepted_traj_file
        self.checkpoint_interval = checkpoint_interval
        self.enable_hybrid_md = enable_hybrid_md
        self.md_move_prob = md_move_prob
        self.md_steps = md_steps
        self.md_timestep_fs = md_timestep_fs
        self.md_ensemble = md_ensemble.lower()
        self.md_friction = md_friction
        self.md_init_momenta = md_init_momenta
        self.md_remove_drift = md_remove_drift
        if not (0.0 <= self.md_move_prob <= 1.0):
            raise ValueError("md_move_prob must be in [0, 1].")
        if self.md_ensemble not in ("nve", "langevin"):
            raise ValueError("md_ensemble must be 'nve' or 'langevin'.")

        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.scan_history = []

        unique_elements = set(self.atoms.get_chemical_symbols())
        if swap_elements is None:
            self.swap_elements = list(unique_elements)
        else:
            self.swap_elements = swap_elements

        self.swap_indices = np.array(
            [
                i
                for i, s in enumerate(self.atoms.get_chemical_symbols())
                if s in self.swap_elements
            ]
        )
        self.swap_index_set = set(self.swap_indices.tolist())
        self._swap_neighbors: Optional[list[np.ndarray]] = None
        self._neighbor_cache_ready = False

        self.atoms.calc = self.calculator
        self.e_old = self.atoms.get_potential_energy()
        self.sweep = 0
        self.accepted_moves = 0
        self.total_moves = 0

        if resume:
            self._load_checkpoint()

    def _resolve_neighbor_backend(self, backend: str) -> str:
        backend_norm = backend.lower()
        if backend_norm not in ("ase", "matscipy", "auto"):
            raise ValueError("neighbor_backend must be 'ase', 'matscipy', or 'auto'.")

        if backend_norm in ("matscipy", "auto"):
            try:
                from matscipy.neighbours import neighbour_list as matscipy_neighbor_list

                self._matscipy_neighbor_list = matscipy_neighbor_list
                if backend_norm == "auto":
                    logger.info("Neighbor backend auto-selected: matscipy")
                return "matscipy"
            except ImportError:
                if backend_norm == "matscipy":
                    raise ImportError(
                        "neighbor_backend='matscipy' requires matscipy to be installed."
                    ) from None
                logger.info("Neighbor backend auto-selected: ase")

        self._matscipy_neighbor_list = None
        return "ase"

    def _neighbor_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.neighbor_backend == "matscipy":
            i_idx, j_idx = self._matscipy_neighbor_list(
                "ij",
                self.atoms,
                self.neighbor_cutoff,
            )
        else:
            i_idx, j_idx = neighbor_list(
                "ij",
                self.atoms,
                cutoff=self.neighbor_cutoff,
                self_interaction=False,
            )
        return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int)

    def _compute_swap_neighbor_lists(self) -> list[np.ndarray]:
        n_atoms = len(self.atoms)
        neighbors_by_atom = [np.empty(0, dtype=int) for _ in range(n_atoms)]

        if len(self.swap_indices) < 2:
            return neighbors_by_atom

        i_idx, j_idx = self._neighbor_pairs()
        if i_idx.size == 0:
            return neighbors_by_atom

        is_swap = np.zeros(n_atoms, dtype=bool)
        is_swap[self.swap_indices] = True
        mask = is_swap[i_idx] & is_swap[j_idx] & (i_idx != j_idx)
        if not np.any(mask):
            return neighbors_by_atom

        filtered_i = i_idx[mask]
        filtered_j = j_idx[mask]

        buckets = [[] for _ in range(n_atoms)]
        for i, j in zip(filtered_i, filtered_j):
            buckets[int(i)].append(int(j))

        for i, neigh in enumerate(buckets):
            if neigh:
                neighbors_by_atom[i] = np.unique(np.asarray(neigh, dtype=int))

        return neighbors_by_atom

    def _invalidate_neighbor_cache(self) -> None:
        self._swap_neighbors = None
        self._neighbor_cache_ready = False

    def _build_neighbor_cache(self) -> None:
        self._swap_neighbors = self._compute_swap_neighbor_lists()
        self._neighbor_cache_ready = True

    def _ensure_neighbor_cache(self) -> None:
        if self.neighbor_cache and not self._neighbor_cache_ready:
            self._build_neighbor_cache()

    def _get_neighbor_candidates(self, idx1: int) -> np.ndarray:
        if self.neighbor_cache:
            self._ensure_neighbor_cache()
            if self._swap_neighbors is None:
                return np.empty(0, dtype=int)
            return self._swap_neighbors[idx1]

        # Recompute with the same builder used by the cached path.
        neighbors_by_atom = self._compute_swap_neighbor_lists()
        return neighbors_by_atom[idx1]

    def _save_checkpoint(self):
        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        state = {
            "atoms": atoms_copy,
            "sweep": self.sweep,
            "e_old": self.e_old,
            "T": self.T,
            "rng_state": self.rng.bit_generator.state,
            "scan_history": self.scan_history,
            "sum_E": self.sum_E,
            "sum_E_sq": self.sum_E_sq,
            "n_samples": self.n_samples,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        with open(self.checkpoint_file, "rb") as f:
            state = pickle.load(f)
        if "atoms" in state:
            self.atoms = state["atoms"]
            self.atoms.calc = self.calculator
        self.T = state.get("T", self.T)
        self.e_old = state.get("e_old", 0.0)
        self.sweep = state.get("sweep", 0)
        self.rng.bit_generator.state = state["rng_state"]
        self.scan_history = state.get("scan_history", [])
        self.sum_E = state.get("sum_E", 0.0)
        self.sum_E_sq = state.get("sum_E_sq", 0.0)
        self.n_samples = state.get("n_samples", 0)
        self._invalidate_neighbor_cache()
        logger.info(f"[{self.T:.0f}K] Resumed from checkpoint.")

    def propose_swap_indices(self) -> Optional[Tuple[int, int]]:
        if len(self.swap_indices) < 2:
            return None

        mode = self.swap_mode
        if mode == "hybrid":
            mode = "neighbor" if self.rng.random() < 0.5 else "global"

        if mode == "global":
            idx1, idx2 = self.rng.choice(self.swap_indices, size=2, replace=False)
        elif mode == "neighbor":
            idx1 = self.rng.choice(self.swap_indices)
            valid = self._get_neighbor_candidates(idx1)
            if valid.size == 0:
                return None
            idx2 = self.rng.choice(valid)
        else:
            return None

        if self.atoms.symbols[idx1] == self.atoms.symbols[idx2]:
            return None
        return idx1, idx2

    def _metropolis_accept(self, delta_e: float, beta: Optional[float] = None) -> bool:
        if delta_e < 0:
            return True
        if beta is None:
            beta = 1.0 / (8.617e-5 * self.T)
        return self.rng.random() < np.exp(-delta_e * beta)

    def _propose_md_move(self) -> Tuple[Optional[Atoms], float]:
        atoms_trial = self.atoms.copy()
        atoms_trial.calc = self.calculator

        if self.md_init_momenta:
            MaxwellBoltzmannDistribution(
                atoms_trial, temperature_K=self.T, rng=self.rng
            )
            if self.md_remove_drift:
                Stationary(atoms_trial)

        dt = self.md_timestep_fs * units.fs
        if self.md_ensemble == "langevin":
            dyn = Langevin(
                atoms_trial,
                timestep=dt,
                temperature_K=self.T,
                friction=self.md_friction,
                rng=self.rng,
            )
        else:
            dyn = VelocityVerlet(atoms_trial, timestep=dt)

        try:
            dyn.run(self.md_steps)
            e_new = self.get_potential_energy(atoms_trial)
        except Exception as exc:
            logger.warning(f"MD trial move failed: {exc}")
            return None, 0.0

        return atoms_trial, e_new - self.e_old

    def relax_structure(
        self, atoms: Atoms, move_ind: Optional[list]
    ) -> tuple[Atoms, bool]:
        if not self.local_relax:
            return super().relax_structure(atoms, move_ind)

        # Local relaxation logic.
        atoms_relax = atoms.copy()

        # Identify active region (swapped atoms plus neighbors).
        # Note: move_ind is [sweep, step], not atom indices.
        # Run() does not pass swapped indices directly, so robust local relaxation would require storing those indices in self.current_swap_indices.

        # Fall back to BaseMC relaxation when robust index tracking is unavailable.
        return super().relax_structure(atoms, move_ind)

    def run(
        self,
        nsweeps: int,
        traj_file: str,
        interval: int = 10,
        sample_interval: int = 1,
        equilibration: int = 0,
    ) -> Dict[str, float]:
        self.traj_file = traj_file

        # Determine trajectory write mode safely.
        if os.path.exists(self.traj_file) and os.path.getsize(self.traj_file) > 0:
            mode = "a"
        else:
            mode = "w"

        self.traj_writer = Trajectory(self.traj_file, mode)

        # Reset accumulators for this run block.
        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.accepted_moves = 0
        self.total_moves = 0
        if self.neighbor_cache:
            self._invalidate_neighbor_cache()

        moves_per_sweep = len(self.swap_indices)

        # Optional start-of-run header can be added here if needed.

        for sweep in range(nsweeps):
            beta = 1.0 / (8.617e-5 * self.T)
            if self.neighbor_cache and self.swap_mode in ("neighbor", "hybrid"):
                self._ensure_neighbor_cache()
            for i in range(moves_per_sweep):
                self.total_moves += 1
                do_md = self.enable_hybrid_md and self.rng.random() < self.md_move_prob
                if do_md:
                    atoms_trial, delta_e = self._propose_md_move()
                    if atoms_trial is None:
                        continue
                    if self._metropolis_accept(delta_e, beta=beta):
                        self.e_old += delta_e
                        self.accepted_moves += 1
                        self.atoms.positions = atoms_trial.positions
                        self.atoms.cell = atoms_trial.cell
                        if self.neighbor_cache:
                            self._invalidate_neighbor_cache()
                    continue

                indices = self.propose_swap_indices()
                if indices is None:
                    continue
                idx1, idx2 = indices

                # Store swap indices for potential local relaxation.
                self.current_swap_indices = [idx1, idx2]

                sym1, sym2 = self.atoms.symbols[idx1], self.atoms.symbols[idx2]
                self.atoms.symbols[idx1], self.atoms.symbols[idx2] = sym2, sym1

                if self.relax:
                    # Use BaseMC relaxation (or local relaxation, if available).
                    atoms_trial = self.atoms.copy()
                    atoms_trial, conv = self.relax_structure(
                        atoms_trial, move_ind=[self.sweep, i]
                    )

                    if conv:
                        e_new = self.get_potential_energy(atoms_trial)
                    else:
                        e_new = 1e9
                else:
                    e_new = self.get_potential_energy(self.atoms)

                delta_e = e_new - self.e_old

                if self._metropolis_accept(delta_e, beta=beta):
                    self.e_old = e_new
                    self.accepted_moves += 1
                    if self.relax:
                        self.atoms.positions = atoms_trial.positions
                        self.atoms.cell = atoms_trial.cell
                        if self.neighbor_cache:
                            self._invalidate_neighbor_cache()
                else:
                    self.atoms.symbols[idx1], self.atoms.symbols[idx2] = sym1, sym2

            self.sweep += 1

            # 1. Sampling.
            if sweep >= equilibration and (sweep + 1) % sample_interval == 0:
                self.sum_E += self.e_old
                self.sum_E_sq += self.e_old**2
                self.n_samples += 1

            # 2. Reporting with temperature tag.
            if (sweep + 1) % interval == 0:
                self.traj_writer.write(self.atoms)
                with open(self.thermo_file, "a") as f:
                    f.write(f"{self.sweep} {self.e_old:.6f}\n")

                acc = (
                    (self.accepted_moves / self.total_moves * 100)
                    if self.total_moves
                    else 0.0
                )
                avg = self.sum_E / self.n_samples if self.n_samples else 0.0
                Cv = 0.0
                if self.n_samples > 1:
                    var = (self.sum_E_sq / self.n_samples) - (avg**2)
                    Cv = var / (8.617e-5 * self.T**2)

                logger.info(
                    f"T={self.T:4.0f}K | {self.sweep:6d} | E: {self.e_old:10.4f} | Avg: {avg:10.4f} | Cv: {Cv:8.4f} | Acc: {acc:4.1f}%"
                )

            # 3. Checkpointing.
            if (
                self.checkpoint_interval > 0
                and self.sweep % self.checkpoint_interval == 0
            ):
                self._save_checkpoint()

        self.traj_writer.close()

        final_avg = self.sum_E / self.n_samples if self.n_samples else self.e_old
        final_Cv = 0.0
        if self.n_samples > 1:
            var = (self.sum_E_sq / self.n_samples) - (final_avg**2)
            final_Cv = var / (8.617e-5 * self.T**2)

        return {
            "T": self.T,
            "energy": final_avg,
            "cv": final_Cv,
            "acceptance": (
                (self.accepted_moves / self.total_moves * 100)
                if self.total_moves
                else 0.0
            ),
        }
