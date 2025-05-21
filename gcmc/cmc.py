import os
import pickle
import logging
import random
import numpy as np
from typing import Optional, Tuple, Any, Literal

from ase import Atoms, Atom
from ase.io import write, read
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS

from .utils import generate_adsorbate_configuration

logger = logging.getLogger("cmc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

class BaseMC:
    """
    Base MC class: shared logic for checkpointing, relaxation, etc.
    """
    def __init__(
        self,
        T: float,
        relax: bool,
        relax_steps: int,
        traj_file: str = "traj.traj",
        accepted_traj_file: str = "unique.traj",
        checkpoint_traj: str = "checkpoint.traj",
        checkpoint_data: str = "checkpoint.pkl",
        checkpoint_interval: int = 1000,
        seed: int = 81,
        fmax: float = 0.05,
    ):
        self.T = T
        self.beta = 1 / (8.617333e-5 * T)
        self.relax = relax
        self.relax_steps = relax_steps
        self.fmax = fmax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.checkpoint_traj = checkpoint_traj
        self.checkpoint_data = checkpoint_data
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed
        self.rng = random.Random(seed)
        self.step = 0
        self.calculator: Optional[Any] = None

    def _save_checkpoint(self) -> None:
        atoms_to_save = self.atoms.copy()
        atoms_to_save.calc = None
        write(self.checkpoint_traj, atoms_to_save)
        state = {
            "step": self.step,
            "e_old": self.e_old,
            "rng_state": self.rng.getstate(),
        }
        with open(self.checkpoint_data, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Checkpoint saved at step {self.step}.")

    def _load_checkpoint(self) -> None:
        self.atoms = read(self.checkpoint_traj)
        with open(self.checkpoint_data, "rb") as f:
            state = pickle.load(f)
        self.step = state.get("step", 0)
        self.e_old = state.get("e_old", None)
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])

    def relax_structure(self, atoms: Atoms) -> Atoms:
        """
        Optionally relax structure using LBFGS optimizer.
        """
        if self.relax:
            atoms_relax = atoms.copy()
            atoms_relax.calc = self.calculator
            dyn = LBFGS(atoms_relax, logfile=None)
            dyn.run(fmax=self.fmax, steps=self.relax_steps)
            return atoms_relax
        else:
            return atoms

class CanonicalMC(BaseMC):
    """
    Canonical Monte Carlo (CMC) with explicit sweeps for fixed-N adsorbates.

    Args:
        substrate_atoms: ASE Atoms object (no adsorbates).
        calculator: ASE calculator.
        T: Temperature (K).
        nsweeps: Number of sweeps (each sweep = N displacement moves).
        relax: Whether to relax each accepted structure.
        relax_steps: Steps for relaxation.
        traj_file, accepted_traj_file: Trajectory output.
        checkpoint_traj, checkpoint_data: Checkpointing.
        checkpoint_interval: Save frequency.
        seed: RNG seed.
        max_trials: Maximum trials per displacement.
        displacement_sigma: Maximum in-plane displacement (Å).
        adsorbate_element: Symbol of adsorbate ("Cu").
        substrate_element: Symbol of substrate ("Ti").
        site_type: "fcc", "hcp", or "atop".
        coverage: ML coverage.
        xy_tol, support_xy_tol, vertical_offset: Geometry parameters.

    Usage:
        cmc = CanonicalMC(...)
        cmc.run(nsweeps=5000)
    """

    def __init__(
        self,
        substrate_atoms: Atoms,
        calculator: Any,
        T: float,
        relax: bool = False,
        relax_steps: int = 20,
        fmax: float = 0.05,
        traj_file: str = "cmc_full.traj",
        accepted_traj_file: str = "cmc_unique.traj",
        checkpoint_traj: str = "cmc_checkpoint.traj",
        checkpoint_data: str = "cmc_checkpoint.pkl",
        checkpoint_interval: int = 1000,
        seed: int = 81,
        max_trials: int = 10,
        displacement_sigma: float = 5.0,
        adsorbate_element: str = "Cu",
        substrate_element: str = "Ti",
        site_type: Literal["atop", "fcc", "hcp"] = "fcc",
        coverage: float = 1.0,
        xy_tol: float = 0.5,
        support_xy_tol: float = 2.0,
        vertical_offset: float = 1.8,
    ) -> None:
        super().__init__(
            T=T,
            relax=relax,
            relax_steps=relax_steps,
            fmax=fmax,
            traj_file=traj_file,
            accepted_traj_file=accepted_traj_file,
            checkpoint_traj=checkpoint_traj,
            checkpoint_data=checkpoint_data,
            checkpoint_interval=checkpoint_interval,
            seed=seed
        )
        self.max_trials = max_trials
        self.displacement_sigma = displacement_sigma
        self.adsorbate_element = adsorbate_element
        self.substrate_element = substrate_element
        self.site_type = site_type
        self.coverage = coverage
        self.xy_tol = xy_tol
        self.support_xy_tol = support_xy_tol
        self.vertical_offset = vertical_offset

        self.calculator = calculator
        self.atoms = generate_adsorbate_configuration(
            atoms=substrate_atoms,
            site_type=site_type,
            element=adsorbate_element,
            coverage=coverage,
            xy_tol=xy_tol,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            substrate_element=substrate_element,
            seed=seed,
        )
        write('init_atoms.xyz', self.atoms)
        self.atoms.calc = calculator

        if os.path.exists(self.checkpoint_traj) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint()
            logger.info(f"Resuming from checkpoint at step {self.step}")
        else:
            if self.relax:
                self.atoms = self.relax_structure(self.atoms)
            self.e_old = self.atoms.get_potential_energy()

    def attempt_displacement(self) -> Optional[Tuple[Atoms, int, np.ndarray, np.ndarray]]:
        """
        Attempt to randomly displace a single adsorbate atom within allowed bounds.
        Returns (Atoms object, index moved, old_xy, new_xy) if move is possible, else None.
        Includes debug info for each trial.
        """
        ads_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.adsorbate_element]
        if not ads_indices:
            logger.warning("No adsorbate atoms found for displacement.")
            return None
        for trial in range(self.max_trials):
            idx = self.rng.choice(ads_indices)
            atom = self.atoms[idx]
            dx = self.rng.uniform(-self.displacement_sigma, self.displacement_sigma)
            dy = self.rng.uniform(-self.displacement_sigma, self.displacement_sigma)
            new_xy = atom.position[:2] + np.array([dx, dy])
            neighbors_z = [
                at.position[2]
                for at in self.atoms
                if np.linalg.norm(at.position[:2] - new_xy) < self.support_xy_tol
            ]
            if not neighbors_z:
                logger.debug(f"[TRIAL {trial}] No support found at xy={new_xy}.")
                continue
            z_max = max(neighbors_z)
            new_pos = np.array([new_xy[0], new_xy[1], z_max + self.vertical_offset])
            too_close = any(
                i != idx and atom2.symbol == self.adsorbate_element
                and np.linalg.norm(new_pos - atom2.position) < self.xy_tol
                for i, atom2 in enumerate(self.atoms)
            )
            if too_close:
                logger.debug(f"[TRIAL {trial}] Rejected: new pos too close to other adsorbate (xy={new_xy}, z={new_pos[2]:.2f}).")
                continue
            logger.debug(f"[TRIAL {trial}] Displace atom {idx} from {atom.position[:2]} to {new_xy}, z={new_pos[2]:.2f}")
            atoms_new = self.atoms.copy()
            atoms_new[idx].position = new_pos
            atoms_new.calc = self.calculator
            return atoms_new, idx, atom.position[:2], new_xy
        logger.debug("All displacement trials failed.")
        return None

    def metropolis_accept(self, e_new: float) -> float:
        delta_e = e_new - self.e_old
        return min(1.0, np.exp(-self.beta * delta_e))

    def run(self, nsweeps: int = 1000) -> None:
        """
        Perform nsweeps MC sweeps (each sweep = N adsorbate displacement attempts).
        Log acceptance rate after each sweep.
        """
        ads_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.adsorbate_element]
        n_ads = len(ads_indices)
        if n_ads == 0:
            logger.error("No adsorbates to move!")
            return

        accepted_traj_mode: str = 'a' if os.path.exists(self.accepted_traj_file) else 'w'
        full_traj_mode: str = 'a' if os.path.exists(self.traj_file) else 'w'
        traj_full = Trajectory(self.traj_file, full_traj_mode)
        header_fmt = "{:>7s} {:>10s} {:>8s} {:>10s} {:>12s} {:>10s} {:>10s} {:>10s}"
        step_fmt   = "{:7d} {:>10s} {:>8s} {:10.2f} {:12.4f} {:10.3f} {:10d} {:10d}"
        logger.info(header_fmt.format("Step", "Move", "Status", "ΔE", "Energy (eV)", "AccRate", "Sweep", "Trial"))

        step_counter = self.step
        for sweep in range(nsweeps):
            n_accepted_sweep = 0
            for trial in range(n_ads):
                result = self.attempt_displacement()
                move_type = "displace"
                if result is None:
                    status = "REJECT"
                    logger.info(step_fmt.format(
                        step_counter, move_type, status, 0.0, self.e_old,
                        n_accepted_sweep/(trial+1), sweep, trial
                    ))
                    traj_full.write(self.atoms, step=step_counter, move=move_type, energy=self.e_old, status=status)
                    step_counter += 1
                    continue
                atoms_new, idx, old_xy, new_xy = result
                atoms_new.calc = self.calculator
                if self.relax:
                    atoms_new = self.relax_structure(atoms_new)
                e_new = atoms_new.get_potential_energy()
                acc_prob = self.metropolis_accept(e_new)
                logger.debug(
                    f"[SWEEP {sweep}][TRIAL {trial}] Atom {idx} from {old_xy} to {new_xy}, "
                    f"e_old={self.e_old:.4f}, e_new={e_new:.4f}, ΔE={e_new-self.e_old:.4f}, p_accept={acc_prob:.3f}"
                )
                if self.rng.random() < acc_prob:
                    self.atoms = atoms_new
                    self.e_old = e_new
                    n_accepted_sweep += 1
                    status = "ACCEPT"
                    # Write only accepted to accepted_traj_file
                    with Trajectory(self.accepted_traj_file, 'a') as traj_acc:
                        traj_acc.write(self.atoms, step=step_counter, move=move_type, energy=self.e_old, status=status)
                else:
                    status = "REJECT"
                logger.info(step_fmt.format(
                    step_counter, move_type, status, e_new-self.e_old, self.e_old,
                    n_accepted_sweep/(trial+1), sweep, trial
                ))
                traj_full.write(self.atoms, step=step_counter, move=move_type, energy=self.e_old, status=status)
                step_counter += 1
            logger.info(
                f"[SWEEP {sweep}] Displacement acceptance: {n_accepted_sweep}/{n_ads} ({n_accepted_sweep/n_ads:.3f})"
            )
            if self.checkpoint_interval and (sweep+1) % self.checkpoint_interval == 0:
                self._save_checkpoint()
        self._save_checkpoint()
        logger.info("CMC completed.")

        traj_full.close()
