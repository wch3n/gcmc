# -*- coding: utf-8 -*-
"""
cmc.py - Canonical Monte Carlo for surface adsorbate systems, with from_clean_surface classmethod
"""

import logging
from typing import Any, Optional, Tuple
import numpy as np
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
        relax: Whether to relax after move
        relax_steps: LBFGS steps per move
        fmax: LBFGS convergence (eV/Angstrom)
        seed: RNG seed
        traj_file: Output file for trajectory
        unique_traj_file: Output file for unique (accepted) structures
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
        z_max_support: float = 3.0,
        vertical_offset: float = 1.8,
        relax: bool = False,
        relax_steps: int = 10,
        fmax: float = 0.05,
        seed: int = 81,
        traj_file: str = "cmc.traj",
        unique_traj_file: str = "cmc_unique.traj",
    ):
        if support_xy_tol is None:
            support_xy_tol = xy_tol
        super().__init__(
            atoms=atoms,
            calculator=calculator,
            adsorbate_element=adsorbate_element,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            neighbor_tol=3.0,
            detach_tol=3.0,
            relax_steps=relax_steps,
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
        self.relax = relax
        self.traj_file = traj_file
        self.unique_traj_file = unique_traj_file

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
        top_layer_element: str = "Ti",  # New explicit argument!
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: str = "fcc",
        xy_tol: float = 0.5,
        support_xy_tol: Optional[float] = 2.5,
        vertical_offset: float = 1.8,
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
            seed=seed,
            **kwargs
        )

    def attempt_displacement(self) -> Optional[Atoms]:
        ads_indices = self.ads_indices
        if len(ads_indices) == 0:
            logger.warning("No adsorbates to displace.")
            return None

        idx = self.rng.choice(ads_indices)
        pos = self.atoms.positions[idx].copy()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()

        # Displacement in xy, wrap to PBC
        delta = self.rng.normal(loc=0.0, scale=self.displacement_sigma, size=2)
        new_xy = pos[:2] + delta

        if any(pbc[:2]):
            xy_matrix = cell[:2, :2]
            frac = np.linalg.solve(xy_matrix.T, new_xy)
            frac = frac % 1.0
            new_xy = np.dot(xy_matrix.T, frac)

        # PBC-aware support check in xy, vectorized
        all_pos = self.atoms.get_positions()
        N = len(all_pos)

        # Pad positions to shape (N,3) with z=0 for xy-distances
        new_xyz = np.zeros((1, 3))
        new_xyz[0, :2] = new_xy[:2]
        all_xyz = np.zeros((N, 3))
        all_xyz[:, :2] = all_pos[:, :2]

        dxy = get_distances(new_xyz, all_xyz, cell=cell, pbc=pbc)[1].flatten()
        dz = pos[2] - all_pos[:, 2]
        mask = (dxy < self.support_xy_tol) & (dz > 0) & (dz < self.z_max_support)
        # Exclude the atom itself
        mask[idx] = False

        if not np.any(mask):
            logger.debug(f"Displacement rejected: no support at new position {new_xy}.")
            return None
        new_z = np.max(all_pos[:, 2][mask]) + self.vertical_offset

        # Only now copy atoms and apply the move
        atoms_new = self.atoms.copy()
        atoms_new[idx].position = [new_xy[0], new_xy[1], new_z]
        return atoms_new

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

        traj = Trajectory(self.traj_file, "w")
        traj_unique = Trajectory(self.unique_traj_file, "w")

        for sweep in range(nsweeps):
            # Dynamic moves per sweep
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
                    if self.relax:
                        atoms_new, converged = self.relax_structure(atoms_new)
                        if not converged:
                            logger.debug("Relaxation did not converge; rejecting move.")
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
                            continue
                        self.atoms = atoms_new
                        self.e_old = e_new
                        self._update_indices()
                        status = "ACCEPT"
                        self.accepted_moves += 1
                        traj_unique.write(self.atoms)
                    else:
                        status = "REJECT"

                acc_rate = (
                    self.accepted_moves / self.total_moves if self.total_moves else 0.0
                )
                logger.info(
                    "{:8d} {:10d} {:>10s} {:12.4f} {:12.4f} {:10.4f}".format(
                        sweep, trial, status, delta_e, self.e_old, acc_rate
                    )
                )
                traj.write(self.atoms)

        traj.close()
        traj_unique.close()
        logger.info("CMC simulation completed.")
