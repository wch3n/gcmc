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
import pickle
from typing import Any, Tuple, Optional, List
from ase import Atoms
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixCartesian
from ase.io import write, read
from ase.geometry import get_distances

logger = logging.getLogger("mc")
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
        detach_tol: float = 3.0,
        relax_steps: int = 10,
        relax_z_only: bool = False,
        fmax: float = 0.05,
        verbose_relax: bool = False,
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

        self.detach_tol = detach_tol
        self.relax_steps = relax_steps
        self.relax_z_only = relax_z_only
        self.fmax = fmax
        self.rng = np.random.default_rng(seed)
        self.verbose_relax = verbose_relax

        # Cache indices for performance
        self._update_indices()
        if self.relax_z_only:
            logger.debug("Only z-coordinates allowed to relax.")
        else:
            logger.debug("Full relaxation (all coordinates free).")

    def _update_indices(self) -> None:
        """Update atom indices for adsorbates, substrate, and functionals."""
        self.ads_indices = [i for i, a in enumerate(self.atoms) if a.symbol == self.adsorbate_element]
        self.sub_indices = [i for i, a in enumerate(self.atoms) if a.symbol in self.substrate_elements]
        self.func_indices = [i for i, a in enumerate(self.atoms) if a.symbol in self.functional_elements]

    def relax_structure(self, atoms: Atoms, move_ind: Optional[list]) -> Tuple[Atoms, bool]:
        """
        Relaxes the given atoms object (in place). 
        If self.relax_z_only is True, only the z coordinates are relaxed.
        Returns (relaxed_atoms, converged)
        """
        atoms_relax = atoms.copy()
        if self.relax_z_only:
            constraints = []
            # Only constrain adsorbate atoms to move in z
            for i in getattr(self, "ads_indices", []):
                constraints.append(FixCartesian(i, mask=[True, True, False]))
            atoms_relax.set_constraint(constraints)

        atoms_relax.calc = self.calculator
        dyn = LBFGS(atoms_relax, logfile=None)
        if self.verbose_relax:
            from ase.io import Trajectory
            traj = Trajectory(f'opt_{move_ind[0]}_{move_ind[1]}.traj', 'w', atoms_relax)
            dyn.attach(traj.write, interval=1)
        converged = dyn.run(fmax=self.fmax, steps=self.relax_steps)
        return atoms_relax, converged

    def has_afloat_adsorbates(
        self,
        atoms: Optional[Atoms] = None,
        support_xy_tol: float = None, 
        z_max_support: float = None,
    ) -> bool:
        """
        Returns True if there is any adsorbate atom that lacks a physical support beneath it.

        Support = any atom (not self) within support_xy_tol in xy, and at a lower z (dz > 0), but not too far below (dz < z_max_support).
        Uses PBC in xy.

        For every afloat adsorbate, prints index, position, and distances to the closest underlying support atom.

        Args:
            atoms: Optional Atoms object (default: self.atoms).
            support_xy_tol: Maximum xy distance to define 'support' (Å).
            z_max_support: Maximum allowed z-separation for support (Å).

        Returns:
            True if any afloat adsorbate found; False otherwise.
        """
        if support_xy_tol is None:
            support_xy_tol = getattr(self, "support_xy_tol", 2.0)
        if z_max_support is None:
            z_max_support = getattr(self, "z_max_support", 0.1)
        if atoms is None:
            atoms = self.atoms

        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        n_atoms = len(atoms)
        adsorbate_element = getattr(self, "adsorbate_element", "Cu")  # fallback default
        ads_indices = [i for i, a in enumerate(atoms) if a.symbol == adsorbate_element]

        if len(ads_indices) == 0:
            logger.debug("No adsorbates found for afloat check.")
            return False

        afloat_found = False

        for ia in ads_indices:
            ads_pos = pos[ia]
            support_indices = [i for i in range(n_atoms) if i != ia]
            support_pos = pos[support_indices]
            deltas, dists = get_distances(ads_pos, support_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)      # In-plane
            dz = ads_pos[2] - support_pos[:, 2]              # Only atoms *below* the adsorbate
            mask = (dxy < support_xy_tol) & (dz > 0) & (dz < z_max_support)
            if not np.any(mask):
                # No support found for this adsorbate
                afloat_found = True
                # Debug: Show all distances and closest candidates
                min_dxy = dxy.min() if dxy.size > 0 else float("nan")
                below_mask = dz > 0
                min_dz = dz[below_mask].min() if np.any(below_mask) else float("nan")
                logger.debug(
                    f"[AFLOAT] Ads {ia} at (x={ads_pos[0]:.2f}, y={ads_pos[1]:.2f}, z={ads_pos[2]:.2f}): "
                    f"min lateral to any support: {min_dxy:.2f} Å), "
                    f"min dz below: {min_dz:.2f} Å"
                )
        if not afloat_found:
            logger.debug("No afloat adsorbates detected.")
        return afloat_found

    def has_detached_functional_groups(
        self,
        atoms: Optional[Atoms] = None,
        detach_tol: float = 3.0,
    ) -> bool:
        """
        Returns True if any functional group atom is farther than detach_tol (in z)
        from the closest substrate atom. Uses PBC in xy for "nearest" checks.
        """
        if atoms is None:
            atoms = self.atoms
        if not hasattr(self, "sub_indices") or not hasattr(self, "func_indices"):
            # Build indices if not present
            self.sub_indices = [i for i, atom in enumerate(atoms) if atom.symbol in self.substrate_elements]
            self.func_indices = [i for i, atom in enumerate(atoms) if self.functional_elements and atom.symbol in self.functional_elements]
        if len(self.func_indices) == 0 or len(self.sub_indices) == 0:
            return False  # No functional groups or substrate

        sub_pos = atoms.get_positions()[self.sub_indices]
        func_pos = atoms.get_positions()[self.func_indices]
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Pad positions to (N,3) if not already (should be, but for safety)
        # Use ASE's get_distances for PBC-aware xy check

        for i, fpos in enumerate(func_pos):
            f_xyz = fpos.reshape(1, 3)
            dists = get_distances(f_xyz, sub_pos, cell=cell, pbc=pbc)[1].flatten()
            min_d = np.min(dists)
            if min_d > detach_tol:
                logger.debug(f"Functional group atom {self.func_indices[i]} ({fpos}) is detached: nearest substrate {min_d:.2f} Å")
                return True
        return False

    def get_non_buried_adsorbate_indices(
        self,
        support_xy_tol: float = None,
        z_tol: float = None,
    ) -> List[int]:
        """
        Returns indices of adsorbates that are *not buried*: i.e., there is
        no other atom (of any type) above them within support_xy_tol in the xy-plane.

        Args:
            support_xy_tol: Max lateral distance (Å) for "above" definition.
            z_tol: Minimum dz to count as "above" (default 0.1 Å).

        Returns:
            List of indices of non-buried adsorbate atoms.
        """
        if support_xy_tol is None:
            support_xy_tol = getattr(self, "support_xy_tol", 2.0)
        if z_tol is None:
            z_tol = getattr(self, "z_tol", 0.1)

        atoms = self.atoms
        ads = [i for i, a in enumerate(atoms) if a.symbol == self.adsorbate_element]
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        non_buried = []
        for ia in ads:
            ads_pos = pos[ia]
            # Exclude self
            other_indices = [i for i in range(len(atoms)) if i != ia]
            other_pos = pos[other_indices]
            deltas, dists = get_distances(ads_pos, other_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)
            dz = other_pos[:, 2] - ads_pos[2]
            # Check if any atom above within lateral distance and above in z
            mask = (dxy < support_xy_tol) & (dz > z_tol)
            if not np.any(mask):
                non_buried.append(ia)
        return non_buried

    def get_potential_energy(self, atoms: Optional[Atoms] = None) -> float:
        """Returns potential energy (calls calculator if necessary)."""
        atoms = atoms if atoms is not None else self.atoms
        atoms.calc = self.calculator
        return atoms.get_potential_energy()

    def _save_checkpoint(self, checkpoint_atoms: str = "checkpoint.traj", checkpoint_data: str = "checkpoint.pkl"):
        """
        Save checkpoint: atoms object and MC state to files.
        """
        # Save atomic structure (no calculator)
        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        write(checkpoint_atoms, atoms_copy)

        # Save MC state
        state = {
            "sweep": getattr(self, "sweep", 0),
            "e_old": getattr(self, "e_old", None),
            "accepted_moves": getattr(self, "accepted_moves", 0),
            "total_moves": getattr(self, "total_moves", 0),
            "rng_state": self.rng.bit_generator.state if hasattr(self, "rng") else None,
             "T": getattr(self, "T", None),
            # add any other important variables here!
        }
        with open(checkpoint_data, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Checkpoint saved at sweep {state['sweep']}.")

    def _load_checkpoint(self, checkpoint_atoms: str = "checkpoint.traj", checkpoint_data: str = "checkpoint.pkl"):
        """
        Load checkpoint: restore atoms object and MC state.
        """
        self.atoms = read(checkpoint_atoms)
        self.atoms.calc = self.calculator
        with open(checkpoint_data, "rb") as f:
            state = pickle.load(f)
        self.sweep = state.get("sweep", 0)
        self.e_old = state.get("e_old", None)
        self.accepted_moves = state.get("accepted_moves", 0)
        self.total_moves = state.get("total_moves", 0)
        if "rng_state" in state and hasattr(self, "rng"):
            self.rng.bit_generator.state = state["rng_state"]
        self.T = state.get("T", self.T)
        logger.debug(f"Resumed from checkpoint at sweep {self.sweep} (T = {self.T})")
