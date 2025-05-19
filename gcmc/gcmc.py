import os
import pickle
import numpy as np
import random
import logging
from typing import List, Optional, Tuple, Any
from ase import Atom, Atoms
from ase.optimize import LBFGS
from ase.io import write, read
from ase.io.trajectory import Trajectory
from ase.neighborlist import NeighborList

logger = logging.getLogger("gcmc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

class StandardSweepGCMC:
    """
    Grand Canonical Monte Carlo for surface clusters with:
      - Standard sweep: N MC attempts per sweep, N = max(#adsorbates, min_moves)
      - Random in-plane insertion (no grid), insert at pillar top (never afloat)
      - Side-exposed deletion and displacement only (never buried)
      - Full support for functional groups, max layers, relaxation, checkpointing
      - General for any adsorbate element
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        mu: float,
        T: float,
        element: str = 'Cu',
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        nsteps: int = 1000,
        relax: bool = False,
        relax_steps: int = 20,
        traj_file: str = 'gcmc_full.traj',
        accepted_traj_file: str = 'gcmc_accepted.traj',
        checkpoint_traj: str = 'gcmc_checkpoint.traj',
        checkpoint_data: str = 'gcmc_checkpoint.pkl',
        checkpoint_interval: int = 1000,
        seed: int = 81,
        tol: float = 1.2,
        xy_tol: float = 1.2,
        z_tol: float = 0.5,
        max_layers: Optional[int] = None,
        layer_spacing: float = 2.0,
        T_anneal: Optional[float] = None,
        nsteps_anneal: Optional[int] = None,
        functional_elements: Optional[Tuple[str, ...]] = None,
        vertical_offset: float = 1.8,
        adsorbate_neighbor_cutoff: float = 2.8,
        min_moves: int = 5,  # Minimum number of moves per sweep (important at low coverage)
        w_insert: float = 0.2,
        w_delete: float = 0.2,
        w_displace: float = 0.6,
        max_trials: int = 10,
    ) -> None:
        self.atoms: Atoms = atoms
        self.calculator: Any = calculator
        self.mu: float = mu
        self.T_prod: float = T
        self.T: float = T
        self.element: str = element
        self.substrate_elements: Tuple[str, ...] = substrate_elements
        self.nsteps: int = nsteps
        self.relax: bool = relax
        self.relax_steps: int = relax_steps
        self.traj_file: str = traj_file
        self.accepted_traj_file: str = accepted_traj_file
        self.checkpoint_traj: str = checkpoint_traj
        self.checkpoint_data: str = checkpoint_data
        self.checkpoint_interval: int = checkpoint_interval
        self.tol: float = tol
        self.xy_tol: float = xy_tol
        self.z_tol: float = z_tol
        self.max_layers: Optional[int] = max_layers
        self.layer_spacing: float = layer_spacing
        self.T_anneal: Optional[float] = T_anneal
        self.nsteps_anneal: Optional[int] = nsteps_anneal
        self.rng: random.Random = random.Random(seed)
        self.seed: int = seed
        self.vertical_offset: float = vertical_offset
        self.adsorbate_neighbor_cutoff: float = adsorbate_neighbor_cutoff
        self.min_moves: int = min_moves
        self.w_insert: float = w_insert
        self.w_delete: float = w_delete
        self.w_displace: float = w_displace
        self.max_trials: int = max_trials

        # --- Auto-detect functional elements if not provided ---
        all_elements = set(atom.symbol for atom in self.atoms)
        if functional_elements is None:
            self.functional_elements = tuple(
                e for e in all_elements if e not in self.substrate_elements and e != self.element
            )
        else:
            self.functional_elements = functional_elements
        logger.info(f"Functional elements: {self.functional_elements}")

        self.attempted_insertions: int = 0
        self.accepted_insertions: int = 0
        self.attempted_deletions: int = 0
        self.accepted_deletions: int = 0
        self.attempted_displacements: int = 0
        self.accepted_displacements: int = 0
        self.step: int = 0

        try:
            self.substrate_z: float = max(
                atom.position[2] for atom in self.atoms
                if atom.symbol in self.substrate_elements
            )
        except ValueError:
            logger.error("No substrate atoms found in the initial structure!")
            raise

        if self.max_layers is not None:
            self.max_z = self.substrate_z + self.max_layers * self.layer_spacing
        else:
            self.max_z = None

        if os.path.exists(self.checkpoint_traj) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint()
            logger.info(f"Resuming from checkpoint at step {self.step}")
        else:
            self.atoms.calc = self.calculator
            self.e_old: float = self.atoms.get_potential_energy()

    def _get_adsorbate_atoms(self) -> List[Atom]:
        return [atom for atom in self.atoms if atom.symbol == self.element]

    def attempt_random_insertion(self) -> Optional[Atoms]:
        """
        Random in-plane insertion: Pick random (x, y) in cell, insert atop nearest pillar.
        """
        for trial in range(self.max_trials):
            cell = self.atoms.get_cell()
            rx, ry = self.rng.random(), self.rng.random()
            xy = rx * cell[0, :2] + ry * cell[1, :2]
            # Find atoms close to (x, y)
            matches = [atom for atom in self.atoms if np.linalg.norm(atom.position[:2] - xy) < self.xy_tol]
            if not matches:
                continue  # No pillar here, skip
            z_top = max(atom.position[2] for atom in matches)
            site = np.array([xy[0], xy[1], z_top + self.vertical_offset])
            if self.max_z is not None and site[2] > self.max_z:
                continue
            too_close = any(
                np.linalg.norm(atom.position - site) < 1.0
                for atom in self.atoms
                if atom.symbol in self.functional_elements or atom.symbol == self.element
            )
            if not too_close:
                atoms_new = self.atoms.copy()
                atoms_new.append(Atom(self.element, site))
                return atoms_new
        logger.debug("Random insertion failed: all trials blocked or not on a pillar.")
        return None

    def get_side_exposed_adsorbate_indices(self) -> List[int]:
        """
        Returns indices of all adsorbate atoms that are side-exposed (not fully encapsulated).
        Uses ASE's NeighborList with PBC, considers edge atoms (fcc: <6 adsorbate neighbors).
        """
        atoms = self.atoms
        symbols = atoms.get_chemical_symbols()
        adsorbate_indices = [i for i, sym in enumerate(symbols) if sym == self.element]
        if not adsorbate_indices:
            return []
        cutoffs = [self.adsorbate_neighbor_cutoff if sym == self.element else 0.0 for sym in symbols]
        nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        removable = []
        for idx in adsorbate_indices:
            indices, offsets = nl.get_neighbors(idx)
            adsorbate_neighbors = [j for j in indices if symbols[j] == self.element]
            if len(adsorbate_neighbors) < 6:
                removable.append(idx)
        return removable

    def get_top_layer_adsorbate_indices(self) -> List[int]:
        """
        Returns indices of adsorbate atoms in the topmost adsorbate layer.
        This is a simpler definition than side-exposed: only atoms within z_tol of the max z among adsorbates.
        """
        adsorbate_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.element]
        if not adsorbate_indices:
            return []
        z_adsorbates = [self.atoms[i].position[2] for i in adsorbate_indices]
        z_max = max(z_adsorbates)
        return [i for i in adsorbate_indices if abs(self.atoms[i].position[2] - z_max) < self.z_tol]

    def attempt_deletion(self) -> Optional[Atoms]:
        side_exposed_indices = self.get_top_layer_adsorbate_indices()
        if not side_exposed_indices:
            logger.debug("No side-exposed adsorbate found for deletion.")
            return None
        atom_idx = self.rng.choice(side_exposed_indices)
        atoms_new = self.atoms.copy()
        del atoms_new[atom_idx]
        return atoms_new

    def attempt_displacement(self) -> Optional[Atoms]:
        side_exposed_indices = self.get_top_layer_adsorbate_indices()
        if not side_exposed_indices:
            logger.debug("No side-exposed adsorbate found for displacement.")
            return None
        for trial in range(self.max_trials):
            cell = self.atoms.get_cell()
            rx, ry = self.rng.random(), self.rng.random()
            xy = rx * cell[0, :2] + ry * cell[1, :2]
            matches = [atom for atom in self.atoms if np.linalg.norm(atom.position[:2] - xy) < self.xy_tol]
            if not matches:
                continue
            z_top = max(atom.position[2] for atom in matches)
            new_pos = np.array([xy[0], xy[1], z_top + self.vertical_offset])
            if self.max_z is not None and new_pos[2] > self.max_z:
                continue
            too_close = any(
                np.linalg.norm(atom.position - new_pos) < 1.0
                for atom in self.atoms
                if atom.symbol in self.functional_elements or atom.symbol == self.element
            )
            if too_close:
                continue
            src_idx = self.rng.choice(side_exposed_indices)
            atoms_new = self.atoms.copy()
            atoms_new[src_idx].position = new_pos
            return atoms_new
        logger.debug("Random displacement failed: all trials blocked or not on a pillar.")
        return None

    def relax_structure(self, atoms: Atoms) -> Atoms:
        if self.relax:
            dyn = LBFGS(atoms, logfile=None)
            dyn.run(fmax=0.05, steps=self.relax_steps)
        return atoms

    def metropolis_accept(self, e_new: float, move_type: str) -> Tuple[float, float]:
        delta_e = e_new - self.e_old
        if move_type == 'insert':
            tau = -self.beta * (delta_e - self.mu)
        elif move_type == 'delete':
            tau = -self.beta * (delta_e + self.mu)
        elif move_type == 'displace':
            tau = -self.beta * delta_e
        else:
            logger.error(f"Unknown move type: {move_type}")
            raise ValueError(f"Unknown move type: {move_type}")
        return np.exp(min(tau, 0)), tau

    def _save_checkpoint(self) -> None:
        atoms_to_save = self.atoms.copy()
        atoms_to_save.calc = None
        write(self.checkpoint_traj, atoms_to_save)
        state = {
            'step': self.step,
            'e_old': self.e_old,
            'attempted_insertions': self.attempted_insertions,
            'accepted_insertions': self.accepted_insertions,
            'attempted_deletions': self.attempted_deletions,
            'accepted_deletions': self.accepted_deletions,
            'attempted_displacements': self.attempted_displacements,
            'accepted_displacements': self.accepted_displacements,
            'rng_state': self.rng.getstate(),
            'T': self.T,
        }
        with open(self.checkpoint_data, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Checkpoint saved at step {self.step}.")

    def _load_checkpoint(self) -> None:
        self.atoms = read(self.checkpoint_traj)
        self.atoms.calc = self.calculator
        with open(self.checkpoint_data, 'rb') as f:
            state = pickle.load(f)
        self.step = state['step']
        self.e_old = state['e_old']
        self.attempted_insertions = state.get('attempted_insertions', 0)
        self.accepted_insertions = state.get('accepted_insertions', 0)
        self.attempted_deletions = state.get('attempted_deletions', 0)
        self.accepted_deletions = state.get('accepted_deletions', 0)
        self.attempted_displacements = state.get('attempted_displacements', 0)
        self.accepted_displacements = state.get('accepted_displacements', 0)
        if 'rng_state' in state:
            self.rng.setstate(state['rng_state'])
        self.T = state.get('T', self.T_prod)

    def run(self, max_moves: Optional[int] = 50) -> None:
        accepted_traj_mode: str = 'a' if os.path.exists(self.accepted_traj_file) else 'w'
        temp_was_logged: bool = False
        full_traj_mode: str = 'a' if os.path.exists(self.traj_file) else 'w'
        traj_full = Trajectory(self.traj_file, full_traj_mode)

        header_fmt = "{:>7s} {:>12s} {:>8s} {:>10s} {:>12s}"
        step_fmt   = "{:7d} {:>12s} {:>8s} {:10.2f} {:12.4f}"
        logger.info(header_fmt.format("Sweep", "Move", "Status", "Tau", "Energy (eV)"))

        with Trajectory(self.accepted_traj_file, accepted_traj_mode) as traj_acc:
            for sweep in range(self.step, self.nsteps):

                # Simulated Annealing if enabled
                if self.T_anneal is not None and self.nsteps_anneal is not None and sweep < self.nsteps_anneal:
                    if not temp_was_logged:
                        logger.info(f"Simulated annealing: T={self.T_anneal} K for first {self.nsteps_anneal} sweeps")
                        temp_was_logged = True
                    self.T = self.T_anneal
                else:
                    if self.T_anneal is not None and self.nsteps_anneal is not None and sweep == self.nsteps_anneal:
                        logger.info(f"Switching to production T={self.T_prod} K at sweep {sweep}")
                    self.T = self.T_prod
                self.beta = 1 / (8.617333e-5 * self.T)

                N_ads = len(self._get_adsorbate_atoms())
                n_moves = max(N_ads, self.min_moves)
                if max_moves is not None:
                    n_moves = min(n_moves, max_moves)

                for m in range(n_moves):
                    # At low coverage, only insertion possible
                    if N_ads == 0:
                        move_type = 'insert'
                    else:
                        move_type = self.rng.choices(
                            ['insert', 'delete', 'displace'],
                            weights=[self.w_insert, self.w_delete, self.w_displace]
                        )[0]

                    atoms_new: Optional[Atoms] = None
                    move_type_log: str = move_type

                    if move_type == 'insert':
                        self.attempted_insertions += 1
                        atoms_new = self.attempt_random_insertion()
                        if atoms_new is None:
                            logger.debug(f"Insertion attempt failed or not possible.")
                            continue
                    elif move_type == 'delete':
                        self.attempted_deletions += 1
                        atoms_new = self.attempt_deletion()
                        if atoms_new is None:
                            logger.debug("Deletion attempt failed or not possible.")
                            continue
                    elif move_type == 'displace':
                        self.attempted_displacements += 1
                        atoms_new = self.attempt_displacement()
                        if atoms_new is None:
                            logger.debug("Displacement attempt failed or not possible.")
                            continue
                    else:
                        logger.warning(f"Unrecognized move type: {move_type}")
                        continue

                    atoms_new.calc = self.calculator
                    if self.relax:
                        atoms_new = self.relax_structure(atoms_new)
                    e_new: float = atoms_new.get_potential_energy()
                    prob, tau = self.metropolis_accept(e_new, move_type)
                    status: str
                    if self.rng.random() < prob:
                        # Move accepted
                        self.atoms = atoms_new
                        self.e_old = e_new
                        if move_type == 'insert':
                            self.accepted_insertions += 1
                        elif move_type == 'delete':
                            self.accepted_deletions += 1
                        elif move_type == 'displace':
                            self.accepted_displacements += 1
                        status = "ACCEPT"
                        logger.info(step_fmt.format(sweep, move_type_log, status, tau, e_new))
                        self.atoms.calc = None
                        traj_acc.write(self.atoms, step=sweep, move=move_type_log, energy=self.e_old, status=status)
                    else:
                        status = "REJECT"
                        logger.info(step_fmt.format(sweep, move_type_log, status, tau, self.e_old))
                    self.atoms.calc = None
                    traj_full.write(self.atoms, step=sweep, move=move_type_log, energy=self.e_old, status=status)

                self.step = sweep + 1
                if (self.checkpoint_interval and (self.step % self.checkpoint_interval == 0)):
                    self._save_checkpoint()

            self._save_checkpoint()
            logger.info("GCMC completed.")
            logger.info(f"Insertions: attempted {self.attempted_insertions}, accepted {self.accepted_insertions}")
            logger.info(f"Deletions: attempted {self.attempted_deletions}, accepted {self.accepted_deletions}")
            logger.info(f"Displacements: attempted {self.attempted_displacements}, accepted {self.accepted_displacements}")

        traj_full.close()
