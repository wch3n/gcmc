# -*- coding: utf-8 -*-
"""
Canonical Monte Carlo for surface adsorbate systems.
"""

import logging
import numpy as np
import os
from typing import Any, Optional, Tuple
from ase import Atoms
from ase.io import write, Trajectory
from ase.geometry import get_distances

from .base import BaseMC
from .utils import generate_adsorbate_configuration

logger = logging.getLogger("mc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


class CMC(BaseMC):
    """
    Canonical Monte Carlo class for fixed-number adsorbate systems.

    Args:
        atoms: ASE Atoms object (with adsorbates)
        calculator: ASE calculator
        adsorbate_element: Symbol of adsorbate (e.g., "Cu")
        substrate_elements: Tuple of substrate elements (e.g., ("Ti", "C"))
        functional_elements: Tuple of functional group elements (optional)
        coverage: ML coverage (e.g., 1.0 = 1 ML)
        site_type: "fcc", "hcp", or "atop"
        displacement_sigma: Width of displacement moves (Angstrom)
        xy_tol: Tolerance for site registry
        support_xy_tol: Lateral cutoff for "support" check
        z_max_support: Max z-separation for "support" (Angstrom)
        vertical_offset: Vertical offset of adsorbate relative to the substrate (Angstrom)
        relax: Whether to relax after move
        relax_steps: LBFGS steps per move
        fmax: LBFGS convergence (eV/Angstrom)
        seed: RNG seed
        traj_file: Output file for trajectory
        accepted_traj_file: Output file for accepted (accepted) structures
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "Cu",
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: str = "fcc",
        displacement_sigma: float = 1.5,
        xy_tol: float = 0.5,
        support_xy_tol: Optional[float] = 2.5,
        z_max_support: float = 3.5,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        relax: bool = False,
        relax_steps: int = 10,
        relax_z_only: bool = False,
        fmax: float = 0.05,
        verbose_relax: bool = False,
        seed: int = 81,
        traj_file: str = "cmc.traj",
        accepted_traj_file: str = "cmc_accepted.traj",
        rejected_traj_file: str = "cmc_rejected.traj",
        attempted_traj_file: str = "cmc_attempted.traj",
        checkpoint_atoms: str = "checkpoint.traj",
        checkpoint_data: str = "checkpoint.pkl",
        checkpoint_interval: int = 100,  # Save every N sweeps.
        resume: bool = False,
    ):
        if support_xy_tol is None:
            support_xy_tol = xy_tol
        super().__init__(
            atoms=atoms,
            calculator=calculator,
            adsorbate_element=adsorbate_element,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            detach_tol=detach_tol,
            relax_steps=relax_steps,
            relax_z_only=relax_z_only,
            fmax=fmax,
            seed=seed,
        )
        self.coverage = coverage
        self.site_type = site_type
        self.displacement_sigma = displacement_sigma
        self.xy_tol = xy_tol
        self.support_xy_tol = support_xy_tol
        self.z_max_support = z_max_support
        self.vertical_offset = vertical_offset
        self.detach_tol = detach_tol
        self.relax = relax
        self.verbose_relax = verbose_relax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.rejected_traj_file = rejected_traj_file
        self.attempted_traj_file = attempted_traj_file
        self.checkpoint_atoms = checkpoint_atoms
        self.checkpoint_data = checkpoint_data
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume

        self.e_old = self.get_potential_energy(self.atoms)
        self._update_indices()
        self.accepted_moves = 0
        self.total_moves = 0


    @classmethod
    def from_clean_surface(
        cls,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "Cu",
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        top_layer_element: str = "Ti",  # Explicit top-layer element.
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: str = "fcc",
        xy_tol: float = 0.5,
        support_xy_tol: Optional[float] = 2.5,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        seed: int = 81,
        **kwargs
    ) -> "CMC":
        """
        Instantiate CMC with generated adsorbate configuration from a clean substrate.
        The initial, rigid structure will be saved as 'cmc_initial.traj'.
        """
        if support_xy_tol is None:
            support_xy_tol = xy_tol

        logger.info("Generating initial adsorbate configuration (rigid) ...")
        atoms_with_ads = generate_adsorbate_configuration(
            atoms=atoms,
            site_type=site_type,
            element=adsorbate_element,
            coverage=coverage,
            xy_tol=xy_tol,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            substrate_element=top_layer_element, 
            seed=seed,
        )
        write("cmc_initial.traj", atoms_with_ads)
        logger.info("Rigid initial structure written to cmc_initial.traj.")

        return cls(
            atoms=atoms_with_ads,
            calculator=calculator,
            adsorbate_element=adsorbate_element,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            coverage=coverage,
            site_type=site_type,
            xy_tol=xy_tol,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            detach_tol=detach_tol,
            seed=seed,
            **kwargs
        )

    def attempt_displacement(self, max_trials: int = 10) -> Optional[Atoms]:
        ads_indices = self.ads_indices
        if len(ads_indices) == 0:
            logger.warning("No adsorbates to displace.")
            return None
        non_buried = self.get_non_buried_adsorbate_indices()
        if not non_buried:
            return None 
        idx = self.rng.choice(non_buried)

        pos = self.atoms.positions[idx].copy()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        tol = self.xy_tol 
        support_xy_tol = self.support_xy_tol

        all_pos = self.atoms.get_positions()
        N = len(all_pos)
        xy_matrix = cell[:2, :2]

        for trial in range(max_trials):
            # Displace in xy and wrap under PBC.
            delta = self.rng.normal(loc=0.0, scale=self.displacement_sigma, size=2)
            new_xy = pos[:2] + delta
            if any(pbc[:2]):
                frac = np.linalg.solve(xy_matrix.T, new_xy)
                frac = frac % 1.0
                new_xy = np.dot(xy_matrix.T, frac)

            # Find local support: atoms within support_xy_tol in xy.
            new_xyz = np.zeros((1, 3))
            new_xyz[0, :2] = new_xy[:2]

            all_xyz = np.zeros_like(all_pos)
            all_xyz[:, :2] = all_pos[:, :2]
            dxy = get_distances(new_xyz, all_xyz, cell=cell, pbc=pbc)[1].flatten()

            support_indices = np.where(dxy < support_xy_tol)[0]
            if len(support_indices) > 0:
                z_max = np.max(all_pos[support_indices, 2])
                new_z = z_max + self.vertical_offset
                logger.debug(
                    f"Trial {trial}: Attempted displacement to xy = ( {new_xy[0]:.3f} {new_xy[1]:.3f} ), "
                    f"z set to {new_z:.3f} above {len(support_indices)} local support"
                )
            else:
                # No support found; keep the original z.
                new_z = pos[2]
                logger.debug(
                    f"Trial {trial}: Attempted displacement to xy=({new_xy[0]:.3f},{new_xy[1]:.3f}), "
                    f"no support found, keeping z={new_z:.3f}"
                )

            # Check proximity to existing atoms.
            trial_pos = np.array([[new_xy[0], new_xy[1], new_z]])
            dists = get_distances(trial_pos, all_pos, cell=cell, pbc=pbc)[1].flatten()
            dists[idx] = np.inf  # Exclude self.

            min_dist = np.min(dists)
            if min_dist >= tol:
                logger.debug(
                    f"Trial {trial}: Proposed position accepted with min_dist={min_dist:.2f}"
                )
                atoms_new = self.atoms.copy()
                atoms_new[idx].position = [new_xy[0], new_xy[1], new_z]
                return atoms_new
            else:
                logger.debug(
                    f"Trial {trial}: Rejected due to proximity; min_dist={min_dist:.2f} "
                    f"(tol={tol:.2f}), retrying."
                )

        logger.debug(f"Displacement failed after {max_trials} trials for atom {idx}.")
        return None

    def metropolis_accept(self, e_new: float, T: float = 300.0) -> float:
        """Return acceptance probability for Metropolis criterion at temperature T (default 300K)."""
        delta_e = e_new - self.e_old
        try:
            prob = min(1.0, np.exp(-delta_e / (8.617333e-5 * T)))
        except OverflowError:
            prob = 0.0
        return prob

    def run(
        self,
        nsweeps: int = 100,
        trials_per_sweep: Optional[int] = None,
        T: float = 300.0,
    ) -> None:
        """
        Run canonical MC simulation.
        Args:
            nsweeps: Number of MC sweeps (outer loop)
            trials_per_sweep: Number of attempted moves per sweep (if None, uses max(#adsorbates, 5))
            T: Temperature (K) for Metropolis criterion
        """
        from ase.io import Trajectory

        logger.info(
            "{:>8s} {:>10s} {:>10s} {:>12s} {:>12s} {:>10s}".format(
                "Sweep", "Trial", "Status", "DeltaE", "Energy", "AccRate"
            )
        )

        traj = Trajectory(self.traj_file, "a")
        traj_accepted = Trajectory(self.accepted_traj_file, "a")
        traj_rejected = Trajectory(self.rejected_traj_file, "a")
        traj_attempted = Trajectory(self.attempted_traj_file, "a")
    
        self.T = T
        if self.resume and os.path.exists(self.checkpoint_atoms) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint(self.checkpoint_atoms, self.checkpoint_data)
            if T is not None:
                logger.debug(f"Overriding checkpoint with T = {T}")
                self.T = T
        else:
            self.sweep = 0  # Ensure sweep starts from zero when not resuming.

        for sweep in range(self.sweep, nsweeps):
            # Dynamic moves per sweep.
            if trials_per_sweep is None:
                n_ads = len(self.ads_indices)
                moves_this_sweep = max(n_ads, 5)
            else:
                moves_this_sweep = trials_per_sweep

            for trial in range(moves_this_sweep):
                self.total_moves += 1
                atoms_new = self.attempt_displacement()
                if atoms_new is None:
                    status = "REJECT"
                    delta_e = 0.0
                else:
                    traj_attempted.write(atoms_new)
                    if self.relax:
                        atoms_new, converged = self.relax_structure(atoms_new, [sweep, trial])
                        if not converged:
                            logger.debug(f"Relaxation did not converge after {self.relax_steps} steps")
                        if self.has_detached_functional_groups(atoms_new, detach_tol=self.detach_tol):
                            logger.debug("Functional group detached after relaxation; rejecting move.")
                            status = "REJECT"
                            delta_e = 0.0
                            continue
                    e_new = self.get_potential_energy(atoms_new)
                    prob = self.metropolis_accept(e_new, T)
                    delta_e = e_new - self.e_old

                    if prob > self.rng.random():
                        if self.has_afloat_adsorbates(
                            atoms_new,
                            support_xy_tol=self.support_xy_tol,
                            z_max_support=self.z_max_support,
                        ):
                            logger.debug("Afloat adsorbate found; move rejected.")
                            status = "REJECT"
                            traj_rejected.write(atoms_new)
                            continue
                        self.atoms = atoms_new
                        self.e_old = e_new
                        self._update_indices()
                        status = "ACCEPT"
                        self.accepted_moves += 1
                        traj_accepted.write(self.atoms)
                    else:
                        status = "REJECT"
                        traj_rejected.write(self.atoms)

                acc_rate = (
                    self.accepted_moves / self.total_moves if self.total_moves else 0.0
                )
                logger.info(
                    "{:8d} {:10d} {:>10s} {:12.4f} {:12.4f} {:10.4f}".format(
                        sweep, trial, status, delta_e, self.e_old, acc_rate
                    )
                )
                traj.write(self.atoms)

            # Save checkpoint.
            self.sweep = sweep + 1
            if self.checkpoint_interval and (self.sweep % self.checkpoint_interval == 0):
                self._save_checkpoint(self.checkpoint_atoms, self.checkpoint_data)

        # Save a final checkpoint at the end.
        self._save_checkpoint(self.checkpoint_atoms, self.checkpoint_data)
        traj.close()
        traj_accepted.close()
        traj_rejected.close()
        traj_attempted.close()
        logger.info("CMC simulation completed.")
