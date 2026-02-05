"""
BaseMC class for MC simulation of surface/adsorbate systems.
Ensemble-neutral utilities for canonical and grand canonical MC.

Provides:
- Logging and reproducibility
- Relaxation (Fixed and Variable Cell)
- PBC-aware detection of unsupported (afloat) adsorbates
- Functional group detachment detection
"""

import numpy as np
import logging
import pickle
import os
from typing import Any, Tuple, Optional, List
from ase import Atoms
from ase.optimize import LBFGS
from ase.constraints import FixCartesian

# Safe import for ExpCellFilter (Moved in ASE 3.23.0)
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from ase.io import write, read
from ase.geometry import get_distances

# --- LOGGING SETUP ---
logger = logging.getLogger("mc")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


class LoggerStream:
    """
    Acts as a file-like object to redirect ASE optimizer output
    into the standard Python logger at the DEBUG level.
    """

    def __init__(self, logger, level=logging.DEBUG):
        self.logger = logger
        self.level = level

    def write(self, message):
        # ASE writes newlines that we don't want in logs, so we strip them
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


class BaseMC:
    """
    BaseMC provides:
        - Logging setup
        - RNG seeding
        - Structural relaxation (Atomic positions + Optional Cell)
        - PBC-aware detection of unsupported (afloat) adsorbates
        - Functional group detachment detection
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "Cu",
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        functional_elements: Optional[Tuple[str, ...]] = None,
        detach_tol: float = 3.0,
        relax_steps: int = 100,
        relax_z_only: bool = False,
        relax_cell: bool = False,
        fmax: float = 0.05,
        verbose_relax: bool = False,
        seed: int = 81,
        traj_file: str = "mc_trajectory.traj",
        thermo_file: str = "thermo.dat",
        checkpoint_file: str = "checkpoint.pkl",
        checkpoint_interval: int = 100,
        **kwargs,
    ):
        self.atoms: Atoms = atoms
        self.calculator = calculator
        self.adsorbate_element = adsorbate_element
        self.substrate_elements = substrate_elements

        # Auto-detect functional elements if not provided
        all_elements = set(atom.symbol for atom in self.atoms)
        if functional_elements is None:
            self.functional_elements = tuple(
                e
                for e in all_elements
                if e not in self.substrate_elements and e != self.adsorbate_element
            )
        else:
            self.functional_elements = functional_elements

        # Relaxation parameters
        self.detach_tol = detach_tol
        self.relax_steps = relax_steps
        self.relax_z_only = relax_z_only
        self.relax_cell = relax_cell
        self.fmax = fmax
        self.verbose_relax = verbose_relax

        # RNG & IO
        self.rng = np.random.default_rng(seed)
        self.traj_file = traj_file
        self.thermo_file = thermo_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval

        # Cache indices for performance
        self._update_indices()

        # Attach calculator immediately
        self.atoms.calc = self.calculator

        # Log configuration
        if self.relax_z_only:
            logger.debug("Relaxation: Z-coordinates only.")
        elif self.relax_cell:
            logger.debug("Relaxation: Full (Variable Cell).")
        else:
            logger.debug("Relaxation: Full (Fixed Cell).")

    def _update_indices(self) -> None:
        """Update atom indices for adsorbates, substrate, and functionals."""
        self.ads_indices = [
            i for i, a in enumerate(self.atoms) if a.symbol == self.adsorbate_element
        ]
        self.sub_indices = [
            i for i, a in enumerate(self.atoms) if a.symbol in self.substrate_elements
        ]
        self.func_indices = [
            i for i, a in enumerate(self.atoms) if a.symbol in self.functional_elements
        ]

    def relax_structure(
        self, atoms: Atoms, move_ind: Optional[list] = None
    ) -> Tuple[Atoms, bool]:
        """
        Relaxes the given atoms object (in place).

        Modes:
          1. relax_z_only=True: Only Z coords move (Fixed Cell).
          2. relax_cell=True:   Positions AND Cell move (Variable Cell).
             * Handles 2D Slabs (Vacuum protection) automatically.
          3. Default:           All positions move (Fixed Cell).

        Output is piped to logger at DEBUG level.
        """
        atoms_relax = atoms.copy()
        atoms_relax.calc = self.calculator

        # --- MODE 1: Z-Only Relaxation (Fixed Cell) ---
        if self.relax_z_only:
            constraints = []
            # Only constrain adsorbate atoms to move in z
            for i in getattr(self, "ads_indices", []):
                constraints.append(FixCartesian(i, mask=[True, True, False]))
            atoms_relax.set_constraint(constraints)
            target = atoms_relax

        # --- MODE 2: Variable Cell Relaxation (Crucial for Alloys) ---
        elif self.relax_cell:
            cell = atoms_relax.get_cell()
            # Heuristic: If Z-vector > 20.0 A, assume it's a slab/2D material with vacuum
            is_slab = cell[2, 2] > 20.0

            if is_slab:
                # Relax in-plane (xx, yy, xy). Fix out-of-plane (zz, xz, yz).
                # Mask: 1=Relax, 0=Fix. Order: [xx, yy, zz, yz, xz, xy]
                mask = [1, 1, 0, 0, 0, 1]
                target = ExpCellFilter(atoms_relax, mask=mask)
            else:
                # Full 3D relaxation for bulk
                target = ExpCellFilter(atoms_relax)

        # --- MODE 3: Full Coordinate Relaxation (Fixed Cell) ---
        else:
            target = atoms_relax

        # --- OPTIMIZATION WITH LOGGING ---
        # Route output to logger.debug
        log_stream = LoggerStream(logger, logging.DEBUG)

        dyn = LBFGS(target, logfile=log_stream)

        if self.verbose_relax and move_ind:
            from ase.io import Trajectory

            traj = Trajectory(f"opt_{move_ind[0]}_{move_ind[1]}.traj", "w", atoms_relax)
            dyn.attach(traj.write, interval=1)

        try:
            converged = dyn.run(fmax=self.fmax, steps=self.relax_steps)
        except Exception as e:
            logger.warning(f"Relaxation failed: {e}")
            converged = False

        return atoms_relax, converged

    def has_afloat_adsorbates(
        self,
        atoms: Optional[Atoms] = None,
        support_xy_tol: float = None,
        z_max_support: float = None,
    ) -> bool:
        """
        Returns True if there is any adsorbate atom that lacks a physical support beneath it.
        """
        if support_xy_tol is None:
            support_xy_tol = getattr(self, "support_xy_tol", 2.0)
        if z_max_support is None:
            z_max_support = getattr(self, "z_max_support", 2.5)
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

        for ia in ads_indices:
            ads_pos = pos[ia]
            support_indices = [i for i in range(n_atoms) if i != ia]
            support_pos = pos[support_indices]
            deltas, dists = get_distances(ads_pos, support_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)  # In-plane
            dz = ads_pos[2] - support_pos[:, 2]  # Only atoms *below* the adsorbate

            lateral_mask = dxy < support_xy_tol
            dz_lateral = dz[lateral_mask]
            support_mask = (dz_lateral > 0) & (dz_lateral < z_max_support)
            if not np.any(support_mask):
                if dz_lateral.size > 0 and np.any(dz_lateral > 0):
                    min_dz = dz_lateral[dz_lateral > 0].min()
                    logger.debug(
                        f"[AFLOAT] Ads {ia}: No support. Closest dz={min_dz:.2f} A"
                    )
                else:
                    logger.debug(f"[AFLOAT] Ads {ia}: No atoms within lateral cutoff.")
                return True
        return False

    def has_detached_functional_groups(
        self,
        atoms: Optional[Atoms] = None,
        detach_tol: float = 3.0,
    ) -> bool:
        """
        Returns True if any functional group atom is farther than detach_tol (in z)
        from the closest substrate atom.
        """
        if atoms is None:
            atoms = self.atoms
        if not hasattr(self, "sub_indices") or not hasattr(self, "func_indices"):
            self._update_indices()

        if len(self.func_indices) == 0 or len(self.sub_indices) == 0:
            return False

        sub_pos = atoms.get_positions()[self.sub_indices]
        func_pos = atoms.get_positions()[self.func_indices]
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        for i, fpos in enumerate(func_pos):
            f_xyz = fpos.reshape(1, 3)
            dists = get_distances(f_xyz, sub_pos, cell=cell, pbc=pbc)[1].flatten()
            min_d = np.min(dists)
            if min_d > detach_tol:
                logger.debug(
                    f"Functional group atom {self.func_indices[i]} detached: dist {min_d:.2f} A"
                )
                return True
        return False

    def get_non_buried_adsorbate_indices(
        self,
        support_xy_tol: float = None,
        z_tol: float = None,
    ) -> List[int]:
        """
        Returns indices of adsorbates that are not buried.
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
            other_indices = [i for i in range(len(atoms)) if i != ia]
            other_pos = pos[other_indices]
            deltas, dists = get_distances(ads_pos, other_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)
            dz = other_pos[:, 2] - ads_pos[2]

            mask = (dxy < support_xy_tol) & (dz > z_tol)
            if not np.any(mask):
                non_buried.append(ia)
        return non_buried

    def get_potential_energy(self, atoms: Optional[Atoms] = None) -> float:
        """Returns potential energy (calls calculator if necessary)."""
        target = atoms if atoms is not None else self.atoms
        if target.calc is None:
            target.calc = self.calculator
        return target.get_potential_energy()

    def _save_checkpoint(self):
        """
        Save checkpoint: atoms object and MC state to files.
        Uses unique filename based on trajectory to allow parallel replicas.
        """
        # Save atomic structure
        # e.g., 'replica_300K.traj' -> 'replica_300K_checkpoint.traj'
        chk_traj = self.traj_file.replace(".traj", "_checkpoint.traj")

        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        write(chk_traj, atoms_copy)

        # Save MC state
        state = {
            "sweep": getattr(self, "sweep", 0),
            "e_old": getattr(self, "e_old", None),
            "accepted_moves": getattr(self, "accepted_moves", 0),
            "total_moves": getattr(self, "total_moves", 0),
            "rng_state": self.rng.bit_generator.state if hasattr(self, "rng") else None,
            "T": getattr(self, "T", None),
            # SGCMC Stats preservation (if present)
            "sum_E": getattr(self, "sum_E", 0.0),
            "sum_E_sq": getattr(self, "sum_E_sq", 0.0),
            "n_samples": getattr(self, "n_samples", 0),
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Checkpoint saved at sweep {state['sweep']}.")

    def _load_checkpoint(self):
        """
        Load checkpoint: restore atoms object and MC state.
        """
        if not os.path.exists(self.checkpoint_file):
            return

        # 1. Restore Atoms
        chk_traj = self.traj_file.replace(".traj", "_checkpoint.traj")
        if os.path.exists(chk_traj):
            self.atoms = read(chk_traj)

        self.atoms.calc = self.calculator

        # 2. Restore State
        with open(self.checkpoint_file, "rb") as f:
            state = pickle.load(f)

        self.sweep = state.get("sweep", 0)
        self.e_old = state.get("e_old", None)
        self.accepted_moves = state.get("accepted_moves", 0)
        self.total_moves = state.get("total_moves", 0)
        self.T = state.get("T", self.T)

        if "rng_state" in state and hasattr(self, "rng"):
            self.rng.bit_generator.state = state["rng_state"]

        # Restore SGCMC stats if they exist
        if "sum_E" in state:
            self.sum_E = state["sum_E"]
            self.sum_E_sq = state["sum_E_sq"]
            self.n_samples = state["n_samples"]

        logger.info(f"Resumed from sweep {self.sweep}")
