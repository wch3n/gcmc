# -*- coding: utf-8 -*-
"""
BaseMC class for MC simulation of surface/adsorbate systems.
Ensemble-neutral utilities for canonical and grand canonical MC.

Provides:
- Logging and reproducibility
- Relaxation and energy routines
- PBC-aware detection of unsupported (afloat) adsorbates
- Functional group detachment detection
"""

import numpy as np
import logging
from typing import Any, Tuple, Optional
from ase import Atoms
from ase.optimize import LBFGS

logger = logging.getLogger("basemc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

class BaseMC:
    """
    BaseMC provides:
        - Logging setup
        - RNG seeding
        - Structural relaxation and potential energy
        - PBC-aware detection of unsupported (afloat) adsorbates
        - Functional group detachment detection

    Variables:
        atoms: Working ASE Atoms object
        calculator: ASE calculator for energy/force
        adsorbate_element: Symbol of adsorbate (e.g., "Cu")
        substrate_elements: Substrate elements (e.g., ("Ti", "C"))
        functional_elements: Functional group elements (can be None)
        neighbor_tol: Tolerance for neighbor distance
        detach_tol: Height tolerance for functional group detachment
        relax_steps: Steps for relaxation
        fmax: Convergence threshold for relaxation
        rng: NumPy random generator
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "Cu",
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        functional_elements: Optional[Tuple[str, ...]] = None,
        neighbor_tol: float = 3.0,
        detach_tol: float = 3.0,
        relax_steps: int = 10,
        fmax: float = 0.05,
        seed: int = 81,
    ):
        self.atoms: Atoms = atoms
        self.calculator = calculator
        self.adsorbate_element = adsorbate_element
        self.substrate_elements = substrate_elements

        # Auto-detect functional elements if not provided
        all_elements = set(atom.symbol for atom in self.atoms)
        if functional_elements is None:
            self.functional_elements = tuple(
                e for e in all_elements if e not in self.substrate_elements and e != self.adsorbate_element
            )
        else:
            self.functional_elements = functional_elements

        self.neighbor_tol = neighbor_tol
        self.detach_tol = detach_tol
        self.relax_steps = relax_steps
        self.fmax = fmax
        self.rng = np.random.default_rng(seed)

        # Cache indices for performance
        self._update_indices()

    def _update_indices(self) -> None:
        """Update atom indices for adsorbates, substrate, and functionals."""
        self.ads_indices = [i for i, a in enumerate(self.atoms) if a.symbol == self.adsorbate_element]
        self.sub_indices = [i for i, a in enumerate(self.atoms) if a.symbol in self.substrate_elements]
        self.func_indices = [i for i, a in enumerate(self.atoms) if a.symbol in self.functional_elements]

    def relax_structure(self, atoms: Atoms) -> Tuple[Atoms, bool]:
        """
        Relaxes the given atoms object (in place). Returns (relaxed_atoms, converged).
        """
        atoms.calc = self.calculator
        dyn = LBFGS(atoms, logfile=None)
        converged = dyn.run(fmax=self.fmax, steps=self.relax_steps)
        return atoms, converged

    def has_afloat_adsorbates(
        self,
        atoms: Optional[Atoms] = None,
        support_xy_tol: float = 2.0,
        z_max_support: float = 3.0,
    ) -> bool:
        """
        Returns True if there is any adsorbate (Cu) that has no physical support beneath it:
        - Support = any atom (not self) within xy_tol laterally,
                    at a z coordinate LOWER than the adsorbate (dz > 0) and
                    dz < z_max_support (not too far below).
        Uses PBC in xy.
        Args:
            atoms: Optional Atoms object (default: self.atoms).
            support_xy_tol: float, max xy distance for "support" (Å).
            z_max_support: float, max z separation for "support" (Å).
        Returns:
            True if any unsupported/afloat adsorbate is found; False otherwise.
        """
        if atoms is None:
            atoms = self.atoms

        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        n_atoms = len(atoms)
        ads_indices = [i for i, a in enumerate(atoms) if a.symbol == self.adsorbate_element]

        if len(ads_indices) == 0:
            return False

        for ia in ads_indices:
            xy = pos[ia, :2]
            z = pos[ia, 2]
            # Potential supports: all atoms except self
            support_indices = [i for i in range(n_atoms) if i != ia]
            support_pos = pos[support_indices, :2]
            support_z = pos[support_indices, 2]

            dx = support_pos[:, 0] - xy[0]
            dy = support_pos[:, 1] - xy[1]
            # Minimum image for PBC in xy
            if pbc[0]:
                dx -= cell[0, 0] * np.round(dx / cell[0, 0])
            if pbc[1]:
                dy -= cell[1, 1] * np.round(dy / cell[1, 1])
            dxy = np.sqrt(dx ** 2 + dy ** 2)
            dz = z - support_z
            # Support must be below (dz > 0), but not too far below (dz < z_max_support), and close in xy
            mask = (dxy < support_xy_tol) & (dz > 0) & (dz < z_max_support)
            if not np.any(mask):
                logger.debug(f"Adsorbate {ia} is unsupported/afloat at z={z:.2f}")
                return True
        return False

    def has_detached_functional_groups(
        self, 
        atoms: Optional[Atoms] = None, 
        z_tol: Optional[float] = None
    ) -> bool:
        """
        Checks if any functional group atom is 'detached' from the substrate:
        - If the minimum z-distance between any functional atom and substrate atom
          is larger than z_tol (default: self.detach_tol).
        Returns True if any functional group is detached.
        """
        if atoms is None:
            atoms = self.atoms
        z_tol = self.detach_tol if z_tol is None else z_tol
        func_indices = [i for i, a in enumerate(atoms) if a.symbol in self.functional_elements]
        sub_indices = [i for i, a in enumerate(atoms) if a.symbol in self.substrate_elements]
        if not func_indices or not sub_indices:
            return False  # No functionals or no substrate
        func_z = atoms.get_positions()[func_indices, 2]
        sub_z = atoms.get_positions()[sub_indices, 2]
        min_dist = np.min(np.abs(func_z[:, None] - sub_z[None, :]))
        logger.debug(f"Min z-dist (functional-substrate): {min_dist:.2f} Å (tol: {z_tol})")
        return min_dist > z_tol

    def get_potential_energy(self, atoms: Optional[Atoms] = None) -> float:
        """Returns potential energy (calls calculator if necessary)."""
        atoms = atoms if atoms is not None else self.atoms
        atoms.calc = self.calculator
        return atoms.get_potential_energy()
