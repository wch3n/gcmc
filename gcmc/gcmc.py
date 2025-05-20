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
from scipy.spatial import cKDTree

logger = logging.getLogger("gcmc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

class GCMC:
    """
    General Grand Canonical Monte Carlo (GCMC) for surface adsorption,
    enforcing 'support' for all moves (no floating atoms, no overlaps).

    Parameters (attributes are the same as the __init__ args):
    ----------------------------------------------------------
    atoms: ASE Atoms object (initial slab with or without functionals)
    calculator: ASE-compatible calculator
    mu: Chemical potential (eV) for adsorbate
    T: Simulation temperature (K)
    element: Adsorbate species symbol (e.g. 'Cu')
    nsteps: Number of MC sweeps
    relax: Whether to relax after each move
    relax_steps: Max steps for relaxation
    traj_file, unique_traj_file: Trajectory outputs
    checkpoint_traj, checkpoint_data: Restart files
    checkpoint_interval: Interval between checkpoints
    seed: Random seed
    tol: Minimal allowed distance between adsorbates/functionals (Å)
    xy_tol: In-plane tolerance for 'support' logic (Å)
    z_tol: z-tolerance for defining exposed (top) adsorbates (Å)
    max_layers, layer_spacing: Restrict max adsorbate layers (optional)
    T_anneal, nsteps_anneal: Simulated annealing controls (optional)
    vertical_offset: Height offset above support atom for new adsorbate (Å)
    functional_elements: Which elements to consider as functionals (if None, autodetected)
    min_moves: Minimum MC moves per sweep
    w_insert, w_delete, w_displace: Relative weights for move type probabilities
    max_trials: Maximum attempts for insertion/displacement per move
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        mu: float,
        T: float,
        element: str = 'Cu',
        nsteps: int = 1000,
        relax: bool = False,
        relax_steps: int = 20,
        traj_file: str = 'gcmc_full.traj',
        unique_traj_file: str = 'gcmc_unique.traj',
        checkpoint_traj: str = 'gcmc_checkpoint.traj',
        checkpoint_data: str = 'gcmc_checkpoint.pkl',
        checkpoint_interval: int = 1000,
        seed: int = 81,
        tol: float = 1.5,
        xy_tol: float = 1.2,
        z_tol: float = 0.3,
        max_layers: Optional[int] = None,
        layer_spacing: float = 2.0,
        T_anneal: Optional[float] = None,
        nsteps_anneal: Optional[int] = None,
        vertical_offset: float = 1.8,
        functional_elements: Optional[Tuple[str, ...]] = None,
        min_moves: int = 5,
        w_insert: float = 0.1,
        w_delete: float = 0.1,
        w_displace: float = 0.8,
        max_trials: int = 10,
    ) -> None:
        self.atoms: Atoms = atoms
        self.calculator: Any = calculator
        self.mu: float = mu
        self.T_prod: float = T
        self.T: float = T
        self.element: str = element
        self.nsteps: int = nsteps
        self.relax: bool = relax
        self.relax_steps: int = relax_steps
        self.traj_file: str = traj_file
        self.unique_traj_file: str = unique_traj_file
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
        self.vertical_offset: float = vertical_offset
        self.min_moves: int = min_moves
        self.w_insert: float = w_insert
        self.w_delete: float = w_delete
        self.w_displace: float = w_displace
        self.max_trials: int = max_trials
        self.rng: random.Random = random.Random(seed)
        self.seed: int = seed

        # Move counters
        self.attempted_insertions: int = 0
        self.accepted_insertions: int = 0
        self.attempted_deletions: int = 0
        self.accepted_deletions: int = 0
        self.attempted_displacements: int = 0
        self.accepted_displacements: int = 0
        self.step: int = 0

        # Layer capping, if used
        self.init_z_max: float = max(atom.position[2] for atom in self.atoms)
        if self.max_layers is not None:
            self.max_z = self.init_z_max + self.max_layers * self.layer_spacing
        else:
            self.max_z = None

        # Restart logic
        if os.path.exists(self.checkpoint_traj) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint()
            logger.info(f"Resuming from checkpoint at step {self.step}")
        else:
            self.atoms.calc = self.calculator
            self.e_old: float = self.atoms.get_potential_energy()

    def _get_adsorbate_atoms(self) -> List[Atom]:
        """Return all adsorbate atoms currently present."""
        return [atom for atom in self.atoms if atom.symbol == self.element]

    def get_exposed_adsorbate_indices(self) -> List[int]:
        """
        Return indices of 'exposed' adsorbates (those with no other adsorbate above within xy_tol).
        """
        ads_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.element]
        positions = np.array([self.atoms[i].position for i in ads_indices])
        exposed = []
        for idx, pos in zip(ads_indices, positions):
            # Check for any adsorbate above within xy_tol in-plane
            above = [
                True for other_pos in positions
                if (other_pos[2] > pos[2]) and
                   (np.linalg.norm(other_pos[:2] - pos[:2]) < self.xy_tol)
            ]
            if not any(above):
                exposed.append(idx)
        return exposed

    def attempt_random_insertion(self) -> Optional[Atoms]:
        """
        Try to insert a new adsorbate at a random (x, y) above the highest local support atom (any species),
        provided there is support beneath and no overlap (tol) with any existing atom.
        Detailed debug info if insertion fails.
        """
        for trial in range(self.max_trials):
            cell = self.atoms.get_cell()
            rx, ry = self.rng.random(), self.rng.random()
            xy = rx * cell[0, :2] + ry * cell[1, :2]
            supports = [atom for atom in self.atoms if np.linalg.norm(atom.position[:2] - xy) < self.xy_tol]
            if not supports:
                logger.debug(
                    f"Insertion trial {trial}: No support found beneath xy=({xy[0]:.2f}, {xy[1]:.2f})"
                )
                continue  # No support beneath
            z_insert = max(atom.position[2] for atom in supports) + self.vertical_offset
            if self.max_z is not None and z_insert > self.max_z:
                logger.debug(
                    f"Insertion trial {trial}: z={z_insert:.2f} exceeds max_z={self.max_z:.2f} at xy=({xy[0]:.2f}, {xy[1]:.2f})"
                )
                continue
            site = np.array([xy[0], xy[1], z_insert])
            too_close = any(
                np.linalg.norm(atom.position - site) < self.tol
                for atom in self.atoms
            )
            if too_close:
                logger.debug(
                    f"Insertion trial {trial}: Position ({site[0]:.2f}, {site[1]:.2f}, {site[2]:.2f}) too close to another atom."
                )
                continue
            # Successful trial
            atoms_new = self.atoms.copy()
            atoms_new.append(Atom(self.element, site))
            return atoms_new
        logger.debug("Random insertion failed: all trials blocked or no support beneath.")
        return None

    def attempt_deletion(self) -> Optional[Atoms]:
        """
        Attempt to delete a randomly chosen 'exposed' adsorbate (no one above within xy_tol).
        Detailed debug info if deletion fails.
        """
        exposed_indices = self.get_exposed_adsorbate_indices()
        if not exposed_indices:
            logger.debug("Deletion: No exposed adsorbates available for deletion.")
            return None
        atom_idx = self.rng.choice(exposed_indices)
        ads_atom = self.atoms[atom_idx]
        logger.debug(
            f"Attempting to delete adsorbate at idx={atom_idx}, "
            f"position=({ads_atom.position[0]:.2f}, {ads_atom.position[1]:.2f}, {ads_atom.position[2]:.2f})"
        )
        atoms_new = self.atoms.copy()
        del atoms_new[atom_idx]
        return atoms_new

    def attempt_displacement(self) -> Optional[Atoms]:
        """
        Attempt to move a randomly chosen exposed adsorbate to a new valid supported position,
        ensuring no overlaps and the move does not create or leave floating atoms.
        Detailed debug info if displacement fails.
        """
        exposed_indices = self.get_exposed_adsorbate_indices()
        if not exposed_indices:
            logger.debug("Displacement: No exposed adsorbate available for displacement.")
            return None
        for trial in range(self.max_trials):
            cell = self.atoms.get_cell()
            rx, ry = self.rng.random(), self.rng.random()
            xy = rx * cell[0, :2] + ry * cell[1, :2]
            supports = [atom for atom in self.atoms if np.linalg.norm(atom.position[:2] - xy) < self.xy_tol]
            if not supports:
                logger.debug(
                    f"Displacement trial {trial}: No support found beneath xy=({xy[0]:.2f}, {xy[1]:.2f})"
                )
                continue
            z_new = max(atom.position[2] for atom in supports) + self.vertical_offset
            if self.max_z is not None and z_new > self.max_z:
                logger.debug(
                    f"Displacement trial {trial}: z={z_new:.2f} exceeds max_z={self.max_z:.2f} at xy=({xy[0]:.2f}, {xy[1]:.2f})"
                )
                continue
            new_pos = np.array([xy[0], xy[1], z_new])
            too_close = any(
                np.linalg.norm(atom.position - new_pos) < self.tol
                for atom in self.atoms
            )
            if too_close:
                logger.debug(
                    f"Displacement trial {trial}: Position ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}) too close to another atom."
                )
                continue
            src_idx = self.rng.choice(exposed_indices)
            old_pos = self.atoms[src_idx].position
            logger.debug(
                f"Displacement trial {trial}: Moving adsorbate idx={src_idx} "
                f"from ({old_pos[0]:.2f}, {old_pos[1]:.2f}, {old_pos[2]:.2f}) "
                f"to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
            )
            atoms_new = self.atoms.copy()
            atoms_new[src_idx].position = new_pos
            return atoms_new
        logger.debug("Displacement failed: all trials blocked or no support beneath.")
        return None

    def all_adsorbates_supported(self, atoms: Atoms, cutoff: float = 3.5) -> bool:
        """
        Returns True if all adsorbate atoms have at least one neighbor (any atom except itself) within `cutoff` Å.
        Otherwise, returns False (i.e., desorbed/unsupported atom exists).
        """
        pos_all = atoms.positions
        sym_all = atoms.get_chemical_symbols()
        ads_indices = [i for i, s in enumerate(sym_all) if s == self.element]
        if not ads_indices:
            return True

        tree = cKDTree(pos_all)
        for idx in ads_indices:
            # Find all neighbors within cutoff (excluding self)
            neighbor_indices = tree.query_ball_point(pos_all[idx], cutoff)
            neighbor_indices = [j for j in neighbor_indices if j != idx]
            if not neighbor_indices:
                return False  # Found an orphan/desorbed atom
        return True

    def relax_structure(self, atoms: Atoms) -> Tuple[Atoms, bool]:
        """
        Returns (relaxed_atoms, converged).
        Converged is True if relaxation stopped by reaching fmax criterion.
        """
        atoms_relaxed = atoms.copy()
        atoms_relaxed.calc = self.calculator
        dyn = LBFGS(atoms_relaxed, logfile=None)
        converged = dyn.run(fmax=0.05, steps=self.relax_steps)
        return atoms_relaxed, converged

    def metropolis_accept(self, e_new: float, move_type: str) -> Tuple[float, float]:
        """
        Return (acceptance probability, tau argument) for Metropolis criterion.
        """
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
        """Save current structure and state for restart."""
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
        """Restart from previous checkpoint."""
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

    def _acceptance_rate(self, attempted: int, accepted: int) -> float:
        """Helper to compute acceptance rate percentage."""
        return (accepted / attempted * 100) if attempted > 0 else 0.0

    def run(self, max_moves: Optional[int] = 200, log_every: int = 1) -> None:
        """
        Run the GCMC simulation for nsteps sweeps.
        Each sweep = N moves, where N = max(number of adsorbates, min_moves), capped by max_moves if set.
        """
        accepted_traj_mode: str = 'a' if os.path.exists(self.unique_traj_file) else 'w'
        temp_was_logged: bool = False
        full_traj_mode: str = 'a' if os.path.exists(self.traj_file) else 'w'
        traj_full = Trajectory(self.traj_file, full_traj_mode)

        header_fmt = "{:>7s} {:>7s} {:>12s} {:>8s} {:>10s} {:>12s}"
        step_fmt   = "{:7d} {:7d} {:>12s} {:>8s} {:10.2f} {:12.4f}"
        logger.info(header_fmt.format("Sweep", "Trial", "Move", "Status", "Tau", "Energy (eV)"))

        with Trajectory(self.unique_traj_file, accepted_traj_mode) as traj_acc:
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

                    if self.relax:
                        atoms_relaxed, converged = self.relax_structure(atoms_new)
                        # Check for excessive movement
                        disp = np.linalg.norm(atoms_new.get_positions() - atoms_relaxed.get_positions(), axis=1).max()
                        if not converged:
                            logger.debug("Rejected: relaxation did not converge.")
                            continue
                        if disp > 3.0:  # adjust as appropriate
                            logger.debug(f"Rejected: atoms moved too much during relaxation (max disp {disp:.2f} Å).")
                            continue
                        atoms_new = atoms_relaxed  # Accept the relaxed geometry
                        # filter out desorbed configurations
                        if not self.all_adsorbates_supported(atoms_new):
                            logger.debug("Rejected: found desorbed adsorbates after relaxation.")
                            continue  
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
                        logger.info(step_fmt.format(sweep, m, move_type_log, status, tau, e_new))
                        self.atoms.calc = None
                        traj_acc.write(self.atoms, step=sweep, move=move_type_log, energy=self.e_old, status=status)
                    else:
                        status = "REJECT"
                        logger.info(step_fmt.format(sweep, m, move_type_log, status, tau, self.e_old))
                    self.atoms.calc = None
                    traj_full.write(self.atoms, step=sweep, move=move_type_log, energy=self.e_old, status=status)

                if sweep % log_every == 0:
                    msg = (
                        f"Sweep {sweep:5d}: "
                        f"Ins {self.accepted_insertions}/{self.attempted_insertions} "
                        f"({self._acceptance_rate(self.attempted_insertions, self.accepted_insertions):5.1f}%), "
                        f"Del {self.accepted_deletions}/{self.attempted_deletions} "
                        f"({self._acceptance_rate(self.attempted_deletions, self.accepted_deletions):5.1f}%), "
                        f"Disp {self.accepted_displacements}/{self.attempted_displacements} "
                        f"({self._acceptance_rate(self.attempted_displacements, self.accepted_displacements):5.1f}%)"
                    )
                    logger.debug(msg)

                self.step = sweep + 1
                if (self.checkpoint_interval and (self.step % self.checkpoint_interval == 0)):
                    self._save_checkpoint()

        self._save_checkpoint()
        logger.info("GCMC completed.")
        logger.info(
            f"Total trial moves: {self.attempted_insertions + self.attempted_deletions + self.attempted_displacements:6d}, "
            f"Total acceptance: "
            f"Ins {self.accepted_insertions}/{self.attempted_insertions} "
            f"({self._acceptance_rate(self.attempted_insertions, self.accepted_insertions):.1f}%), "
            f"Del {self.accepted_deletions}/{self.attempted_deletions} "
            f"({self._acceptance_rate(self.attempted_deletions, self.accepted_deletions):.1f}%), "
            f"Disp {self.accepted_displacements}/{self.attempted_displacements} "
            f"({self._acceptance_rate(self.attempted_displacements, self.accepted_displacements):.1f}%)"
        )

        traj_full.close()
