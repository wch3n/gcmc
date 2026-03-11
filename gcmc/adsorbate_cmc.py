import logging
import numpy as np
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ase import Atoms
from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from ase.symbols import string2symbols

from .base import BaseMC
from .utils import (
    classify_hollow_sites,
    get_hollow_xy,
    get_toplayer_xy,
)

logger = logging.getLogger("mc")
ADSORBATE_TAG_OFFSET = 1_000_000


def _load_adsorbate_template(
    adsorbate: Optional[Union[str, Atoms]],
    fallback_element: str,
) -> Atoms:
    if adsorbate is None:
        return Atoms(symbols=[fallback_element], positions=[(0.0, 0.0, 0.0)])
    if isinstance(adsorbate, Atoms):
        return adsorbate.copy()
    if isinstance(adsorbate, str):
        if os.path.exists(adsorbate):
            return read(adsorbate)
        symbols = string2symbols(adsorbate)
        if len(symbols) != 1:
            raise ValueError(
                "String adsorbates must be a single chemical symbol or a structure file. "
                "For molecular adsorbates, pass an ASE Atoms template or a file path."
            )
        return Atoms(symbols=symbols, positions=[(0.0, 0.0, 0.0)])
    raise TypeError("adsorbate must be None, a chemical symbol/path string, or ASE Atoms.")


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _site_xy_for_surface(
    atoms: Atoms,
    site_type: str,
    top_layer_element: Optional[str],
    xy_tol: float,
) -> np.ndarray:
    if top_layer_element is None:
        return np.empty((0, 2), dtype=float)
    if site_type == "atop":
        return np.asarray(get_toplayer_xy(atoms, element=top_layer_element), dtype=float)

    atop_xy = get_toplayer_xy(atoms, element=top_layer_element)
    hollow_xy = get_hollow_xy(atop_xy, atoms.get_cell().array)
    return np.asarray(
        classify_hollow_sites(
            atoms,
            hollow_xy,
            element=top_layer_element,
            xy_tol=xy_tol,
            stacking=site_type,
        ),
        dtype=float,
    )


def _place_adsorbate_template(
    atoms: Atoms,
    adsorbate_template: Atoms,
    *,
    anchor_index: int,
    site_xy: np.ndarray,
    coverage: float,
    support_xy_tol: float,
    vertical_offset: float,
    seed: int,
) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms_new = atoms.copy()
    cell = atoms_new.get_cell()
    relative = (
        adsorbate_template.get_positions()
        - adsorbate_template.positions[anchor_index].copy()
    )
    n_sites = len(site_xy)
    if n_sites == 0:
        raise RuntimeError("*** NO REGISTRY SITES AVAILABLE ***")

    tags = np.asarray(atoms_new.get_tags(), dtype=int)
    if len(tags) != len(atoms_new):
        tags = np.zeros(len(atoms_new), dtype=int)

    all_site_indices = np.arange(n_sites)
    n_full_layers = int(np.floor(coverage))
    frac_layer = coverage - n_full_layers
    selected_layers = [all_site_indices for _ in range(n_full_layers)]
    if frac_layer > 1e-8:
        n_partial = int(np.floor(frac_layer * n_sites + 0.5))
        n_partial = min(n_sites, max(0, n_partial))
        if n_partial > 0:
            selected_layers.append(rng.choice(n_sites, n_partial, replace=False))

    group_id = 0
    for layer_indices in selected_layers:
        for site_idx in np.asarray(layer_indices, dtype=int):
            xy = np.asarray(site_xy[site_idx], dtype=float)
            neighbors_z = []
            for atom in atoms_new:
                dxy = atom.position[:2] - xy
                trial_xyz = np.zeros((1, 3))
                trial_xyz[0, :2] = dxy
                mic = get_distances(
                    np.zeros((1, 3)), trial_xyz, cell=cell, pbc=atoms_new.pbc
                )[1].flatten()[0]
                if mic < support_xy_tol:
                    neighbors_z.append(atom.position[2])
            if not neighbors_z:
                raise RuntimeError(
                    f"No support found at site ({xy[0]:.3f}, {xy[1]:.3f}) for adsorbate placement."
                )
            anchor_z = max(neighbors_z) + vertical_offset
            anchor_pos = np.array([xy[0], xy[1], anchor_z], dtype=float)

            group_tag = ADSORBATE_TAG_OFFSET + group_id
            for symbol, rel in zip(adsorbate_template.get_chemical_symbols(), relative):
                atoms_new.append(symbol)
                atoms_new.positions[-1] = anchor_pos + rel
                tags = np.append(tags, group_tag)
            group_id += 1

    atoms_new.set_tags(tags)
    return atoms_new


class AdsorbateCMC(BaseMC):
    """
    Canonical Monte Carlo for fixed-loading adsorbates on a surface.

    This class is designed to match the `AlloyCMC`/`ReplicaExchange` worker
    interface. Molecular adsorbates are handled as rigid groups whose anchor
    atom defines support and site-hop placement.
    """

    def __init__(
        self,
        atoms: Union[Atoms, str],
        calculator: Any,
        T: float = 300.0,
        adsorbate_element: str = "H",
        adsorbate: Optional[Union[str, Atoms]] = None,
        adsorbate_anchor_index: int = 0,
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        functional_elements: Optional[Tuple[str, ...]] = None,
        top_layer_element: Optional[str] = None,
        coverage: Optional[float] = None,
        site_type: str = "fcc",
        move_mode: str = "displacement",
        site_hop_prob: float = 0.5,
        reorientation_prob: float = 0.2,
        displacement_sigma: float = 1.5,
        max_displacement_trials: int = 10,
        max_reorientation_trials: Optional[int] = None,
        rotation_max_angle_deg: float = 25.0,
        min_moves_per_sweep: int = 5,
        xy_tol: float = 0.5,
        support_xy_tol: Optional[float] = None,
        z_max_support: float = 3.5,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        relax: bool = False,
        relax_steps: int = 10,
        relax_z_only: bool = False,
        fmax: float = 0.05,
        verbose_relax: bool = False,
        traj_file: str = "adsorbate_cmc.traj",
        accepted_traj_file: Optional[str] = None,
        rejected_traj_file: Optional[str] = None,
        attempted_traj_file: Optional[str] = None,
        thermo_file: str = "adsorbate_cmc.dat",
        checkpoint_file: str = "adsorbate_cmc.pkl",
        checkpoint_interval: int = 100,
        seed: int = 81,
        resume: bool = False,
        **kwargs,
    ):
        if isinstance(atoms, str):
            self.atoms = read(atoms)
        else:
            self.atoms = atoms.copy()

        self.adsorbate_template = _load_adsorbate_template(adsorbate, adsorbate_element)
        if len(self.adsorbate_template) == 0:
            raise ValueError("adsorbate template must contain at least one atom.")
        if not (0 <= int(adsorbate_anchor_index) < len(self.adsorbate_template)):
            raise ValueError("adsorbate_anchor_index is out of range for the adsorbate template.")
        self.adsorbate_anchor_index = int(adsorbate_anchor_index)
        self.adsorbate_size = len(self.adsorbate_template)
        self.is_molecular_adsorbate = self.adsorbate_size > 1
        self.adsorbate_anchor_symbol = self.adsorbate_template[
            self.adsorbate_anchor_index
        ].symbol
        self.adsorbate_symbols = tuple(self.adsorbate_template.get_chemical_symbols())
        self._adsorbate_symbol_signature = tuple(sorted(self.adsorbate_symbols))
        initial_tags = np.asarray(self.atoms.get_tags(), dtype=int)
        if len(initial_tags) != len(self.atoms):
            initial_tags = np.zeros(len(self.atoms), dtype=int)
        auto_functional_elements = functional_elements
        if auto_functional_elements is None and self.is_molecular_adsorbate:
            tagged_mask = initial_tags >= ADSORBATE_TAG_OFFSET
            if np.any(tagged_mask):
                auto_functional_elements = tuple(
                    sorted(
                        {
                            atom.symbol
                            for i, atom in enumerate(self.atoms)
                            if (not tagged_mask[i])
                            and atom.symbol not in set(substrate_elements)
                        }
                    )
                )

        if support_xy_tol is None:
            support_xy_tol = xy_tol

        super().__init__(
            atoms=self.atoms,
            calculator=calculator,
            adsorbate_element=self.adsorbate_anchor_symbol,
            substrate_elements=substrate_elements,
            functional_elements=(
                auto_functional_elements
                if auto_functional_elements is not None or not self.is_molecular_adsorbate
                else ()
            ),
            detach_tol=detach_tol,
            relax_steps=relax_steps,
            relax_z_only=relax_z_only,
            fmax=fmax,
            verbose_relax=verbose_relax,
            seed=seed,
            traj_file=traj_file,
            thermo_file=thermo_file,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=checkpoint_interval,
            **kwargs,
        )

        move_mode = move_mode.lower()
        if move_mode not in ("displacement", "site_hop", "reorientation", "hybrid"):
            raise ValueError(
                "move_mode must be 'displacement', 'site_hop', 'reorientation', or 'hybrid'."
            )
        if not (0.0 <= site_hop_prob <= 1.0):
            raise ValueError("site_hop_prob must be in [0, 1].")
        if not (0.0 <= reorientation_prob <= 1.0):
            raise ValueError("reorientation_prob must be in [0, 1].")
        if move_mode == "hybrid" and (site_hop_prob + reorientation_prob) > 1.0:
            raise ValueError(
                "For move_mode='hybrid', site_hop_prob + reorientation_prob must be <= 1."
            )
        if rotation_max_angle_deg < 0.0:
            raise ValueError("rotation_max_angle_deg must be >= 0.")

        self.T = T
        self.top_layer_element = (
            top_layer_element
            if top_layer_element is not None
            else (substrate_elements[0] if substrate_elements else None)
        )
        self.coverage = coverage
        self.site_type = site_type
        self.move_mode = move_mode
        self.site_hop_prob = float(site_hop_prob)
        self.reorientation_prob = float(reorientation_prob)
        self.displacement_sigma = displacement_sigma
        self.max_displacement_trials = int(max_displacement_trials)
        self.max_reorientation_trials = (
            int(max_reorientation_trials)
            if max_reorientation_trials is not None
            else int(max_displacement_trials)
        )
        self.rotation_max_angle_rad = np.deg2rad(float(rotation_max_angle_deg))
        self.min_moves_per_sweep = int(min_moves_per_sweep)
        self.xy_tol = xy_tol
        self.support_xy_tol = support_xy_tol
        self.z_max_support = z_max_support
        self.vertical_offset = vertical_offset
        self.relax = relax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.rejected_traj_file = rejected_traj_file
        self.attempted_traj_file = attempted_traj_file
        self.thermo_file = thermo_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self._site_xy: Optional[np.ndarray] = None

        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.accepted_moves = 0
        self.total_moves = 0
        self.sweep = 0

        self.atoms.calc = self.calculator
        self.e_old = self.atoms.get_potential_energy()

        if resume:
            self._load_checkpoint()

    def _group_anchor_local_index(
        self, group: np.ndarray, atoms: Optional[Atoms] = None
    ) -> int:
        if atoms is None:
            atoms = self.atoms
        group_symbols = tuple(atoms[int(i)].symbol for i in group)
        if (
            len(group_symbols) == len(self.adsorbate_symbols)
            and group_symbols == self.adsorbate_symbols
        ):
            return self.adsorbate_anchor_index

        anchor_matches = [
            local_idx
            for local_idx, atom_idx in enumerate(group)
            if atoms[int(atom_idx)].symbol == self.adsorbate_anchor_symbol
        ]
        if len(anchor_matches) == 1:
            return int(anchor_matches[0])

        raise ValueError(
            "Could not determine the molecular adsorbate anchor atom. "
            "Use AdsorbateCMC.from_clean_surface(...) or provide tagged groups "
            "with the same atom ordering as the adsorbate template."
        )

    def _anchor_index_for_group(
        self, group: np.ndarray, atoms: Optional[Atoms] = None
    ) -> int:
        return int(group[self._group_anchor_local_index(group, atoms=atoms)])

    def _current_group_relative_positions(
        self, group: np.ndarray, atoms: Optional[Atoms] = None
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms
        anchor_idx = self._anchor_index_for_group(group, atoms=atoms)
        positions = atoms.get_positions()[np.asarray(group, dtype=int)]
        return positions - atoms.positions[anchor_idx]

    def _adsorbate_groups_for_atoms(self, atoms: Atoms) -> list[np.ndarray]:
        if atoms is self.atoms:
            return [np.asarray(group, dtype=int) for group in self.ads_groups]

        tags = np.asarray(atoms.get_tags(), dtype=int)
        tagged_groups = [
            np.where(tags == tag)[0]
            for tag in sorted(np.unique(tags))
            if tag >= ADSORBATE_TAG_OFFSET
        ]
        if tagged_groups:
            return [np.asarray(group, dtype=int) for group in tagged_groups]

        if self.adsorbate_size == 1:
            return [
                np.asarray([i], dtype=int)
                for i, atom in enumerate(atoms)
                if atom.symbol == self.adsorbate_anchor_symbol
            ]

        raise ValueError(
            "Molecular adsorbate validation requires tagged adsorbate groups."
        )

    def _infer_adsorbate_groups(self) -> list[np.ndarray]:
        tags = np.asarray(self.atoms.get_tags(), dtype=int)
        tagged_groups = [
            np.where(tags == tag)[0]
            for tag in sorted(np.unique(tags))
            if tag >= ADSORBATE_TAG_OFFSET
        ]
        if tagged_groups:
            return [np.asarray(group, dtype=int) for group in tagged_groups]

        excluded = set(self.substrate_elements) | set(self.functional_elements)
        if self.adsorbate_size == 1 and self.adsorbate_anchor_symbol not in excluded:
            return [
                np.asarray([i], dtype=int)
                for i, atom in enumerate(self.atoms)
                if atom.symbol == self.adsorbate_anchor_symbol
            ]

        if set(self.adsorbate_symbols).isdisjoint(excluded):
            candidates = [
                i
                for i, atom in enumerate(self.atoms)
                if atom.symbol in set(self.adsorbate_symbols)
            ]
            if len(candidates) % self.adsorbate_size != 0:
                raise ValueError(
                    "Could not infer molecular adsorbate groups: adsorbate atoms are not "
                    "divisible by the template size. Provide a tagged structure or use "
                    "AdsorbateCMC.from_clean_surface(...)."
                )
            groups = []
            for start in range(0, len(candidates), self.adsorbate_size):
                group = np.asarray(candidates[start : start + self.adsorbate_size], dtype=int)
                signature = tuple(sorted(self.atoms[i].symbol for i in group))
                if signature != self._adsorbate_symbol_signature:
                    raise ValueError(
                        "Could not infer molecular adsorbate groups from atom order. "
                        "Provide a tagged structure or use AdsorbateCMC.from_clean_surface(...)."
                    )
                groups.append(group)
            return groups

        raise ValueError(
            "Could not infer adsorbate groups automatically. For molecular adsorbates "
            "with species overlapping the slab/functionals, use a tagged structure or "
            "AdsorbateCMC.from_clean_surface(...)."
        )

    def _tag_adsorbate_groups(self, groups: list[np.ndarray]) -> None:
        tags = np.asarray(self.atoms.get_tags(), dtype=int)
        if len(tags) != len(self.atoms):
            tags = np.zeros(len(self.atoms), dtype=int)
        for group_id, group in enumerate(groups):
            tags[np.asarray(group, dtype=int)] = ADSORBATE_TAG_OFFSET + group_id
        self.atoms.set_tags(tags)

    def _update_indices(self) -> None:
        groups = self._infer_adsorbate_groups()
        self._tag_adsorbate_groups(groups)
        self.ads_groups = [np.asarray(group, dtype=int) for group in groups]
        self.ads_indices = [int(idx) for group in self.ads_groups for idx in group]
        self.ads_anchor_indices = [
            self._anchor_index_for_group(group) for group in self.ads_groups
        ]

        tags = np.asarray(self.atoms.get_tags(), dtype=int)
        ads_mask = tags >= ADSORBATE_TAG_OFFSET
        self.sub_indices = [
            i
            for i, atom in enumerate(self.atoms)
            if (not ads_mask[i]) and atom.symbol in self.substrate_elements
        ]
        self.func_indices = [
            i
            for i, atom in enumerate(self.atoms)
            if (not ads_mask[i]) and atom.symbol in self.functional_elements
        ]

    @classmethod
    def from_clean_surface(
        cls,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "H",
        adsorbate: Optional[Union[str, Atoms]] = None,
        adsorbate_anchor_index: int = 0,
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        top_layer_element: str = "Ti",
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: str = "fcc",
        xy_tol: float = 0.5,
        support_xy_tol: Optional[float] = None,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        seed: int = 81,
        initial_traj_file: str = "adsorbate_cmc_initial.traj",
        **kwargs,
    ) -> "AdsorbateCMC":
        if support_xy_tol is None:
            support_xy_tol = xy_tol
        if functional_elements is None:
            functional_elements = tuple(
                sorted(
                    {
                        atom.symbol
                        for atom in atoms
                        if atom.symbol not in set(substrate_elements)
                    }
                )
            )

        logger.info("Generating initial adsorbate configuration ...")
        adsorbate_template = _load_adsorbate_template(adsorbate, adsorbate_element)
        if not (0 <= int(adsorbate_anchor_index) < len(adsorbate_template)):
            raise ValueError(
                "adsorbate_anchor_index is out of range for the adsorbate template."
            )
        site_xy = _site_xy_for_surface(
            atoms,
            site_type=site_type,
            top_layer_element=top_layer_element,
            xy_tol=xy_tol,
        )
        atoms_with_ads = _place_adsorbate_template(
            atoms,
            adsorbate_template,
            anchor_index=int(adsorbate_anchor_index),
            site_xy=site_xy,
            coverage=coverage,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            seed=seed,
        )
        write(initial_traj_file, atoms_with_ads)
        logger.info("Initial adsorbate structure written to %s.", initial_traj_file)

        return cls(
            atoms=atoms_with_ads,
            calculator=calculator,
            T=kwargs.pop("T", 300.0),
            adsorbate_element=adsorbate_element,
            adsorbate=adsorbate_template,
            adsorbate_anchor_index=adsorbate_anchor_index,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            top_layer_element=top_layer_element,
            coverage=coverage,
            site_type=site_type,
            xy_tol=xy_tol,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            detach_tol=detach_tol,
            seed=seed,
            **kwargs,
        )

    def _save_checkpoint(self):
        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        state = {
            "atoms": atoms_copy,
            "sweep": self.sweep,
            "e_old": self.e_old,
            "T": self.T,
            "rng_state": self.rng.bit_generator.state,
            "sum_E": self.sum_E,
            "sum_E_sq": self.sum_E_sq,
            "n_samples": self.n_samples,
        }
        with open(self.checkpoint_file, "wb") as handle:
            pickle.dump(state, handle)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        with open(self.checkpoint_file, "rb") as handle:
            state = pickle.load(handle)
        if "atoms" in state:
            self.atoms = state["atoms"]
            self.atoms.calc = self.calculator
        self.sweep = state.get("sweep", 0)
        self.e_old = state.get("e_old", self.e_old)
        self.T = state.get("T", self.T)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.sum_E = state.get("sum_E", 0.0)
        self.sum_E_sq = state.get("sum_E_sq", 0.0)
        self.n_samples = state.get("n_samples", 0)
        self._site_xy = None
        self._update_indices()
        logger.info(f"[{self.T:.0f}K] Resumed adsorbate MC from checkpoint.")

    def _metropolis_accept(self, delta_e: float, beta: Optional[float] = None) -> bool:
        if delta_e < 0.0:
            return True
        if beta is None:
            beta = 1.0 / (8.617333e-5 * self.T)
        return self.rng.random() < np.exp(-delta_e * beta)

    def _moves_per_sweep(self) -> int:
        n_ads = len(self.ads_groups)
        if n_ads == 0:
            return 0
        return max(n_ads, self.min_moves_per_sweep)

    def _candidate_support_z(
        self, new_xy: np.ndarray, exclude_indices: Optional[Sequence[int]] = None
    ) -> Optional[float]:
        all_pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        new_xyz = np.zeros((1, 3))
        new_xyz[0, :2] = new_xy[:2]
        all_xyz = np.zeros_like(all_pos)
        all_xyz[:, :2] = all_pos[:, :2]
        dxy = get_distances(new_xyz, all_xyz, cell=cell, pbc=pbc)[1].flatten()
        if exclude_indices is not None:
            for exclude_index in exclude_indices:
                if 0 <= int(exclude_index) < len(dxy):
                    dxy[int(exclude_index)] = np.inf
        support_indices = np.where(dxy < self.support_xy_tol)[0]
        if len(support_indices) == 0:
            return None
        return float(np.max(all_pos[support_indices, 2]) + self.vertical_offset)

    def _position_is_valid(self, idx: int, trial_pos: np.ndarray) -> bool:
        return self._group_positions_are_valid(
            np.asarray([idx], dtype=int), trial_pos.reshape(1, 3)
        )

    def _group_positions_are_valid(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if atoms is None:
            atoms = self.atoms

        group = np.asarray(group, dtype=int)
        trial_positions = np.asarray(trial_positions, dtype=float)
        if len(group) != len(trial_positions):
            raise ValueError("group and trial_positions must have the same length.")

        all_pos = atoms.get_positions()
        other_indices = np.array(
            [i for i in range(len(atoms)) if i not in set(group.tolist())], dtype=int
        )
        if len(other_indices) == 0:
            return True

        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        dists = get_distances(
            trial_positions,
            all_pos[other_indices],
            cell=cell,
            pbc=pbc,
        )[1]
        return float(np.min(dists)) >= self.xy_tol

    def get_non_buried_adsorbate_indices(
        self,
        support_xy_tol: float = None,
        z_tol: float = None,
    ) -> list[int]:
        if support_xy_tol is None:
            support_xy_tol = getattr(self, "support_xy_tol", 2.0)
        if z_tol is None:
            z_tol = getattr(self, "z_tol", 0.1)

        atoms = self.atoms
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        non_buried_group_ids = []
        for group_id, group in enumerate(self.ads_groups):
            anchor_idx = self.ads_anchor_indices[group_id]
            other_indices = [i for i in range(len(atoms)) if i not in set(group.tolist())]
            if not other_indices:
                non_buried_group_ids.append(group_id)
                continue
            anchor_pos = pos[anchor_idx]
            other_pos = pos[other_indices]
            deltas, _ = get_distances(anchor_pos, other_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)
            dz = other_pos[:, 2] - anchor_pos[2]
            mask = (dxy < support_xy_tol) & (dz > z_tol)
            if not np.any(mask):
                non_buried_group_ids.append(group_id)
        return non_buried_group_ids

    def has_afloat_adsorbates(
        self,
        atoms: Optional[Atoms] = None,
        support_xy_tol: float = None,
        z_max_support: float = None,
    ) -> bool:
        if support_xy_tol is None:
            support_xy_tol = getattr(self, "support_xy_tol", 2.0)
        if z_max_support is None:
            z_max_support = getattr(self, "z_max_support", 2.5)
        if atoms is None:
            atoms = self.atoms

        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        groups = self._adsorbate_groups_for_atoms(atoms)

        if len(groups) == 0:
            return False

        for group in groups:
            anchor_idx = int(group[self._group_anchor_local_index(group, atoms=atoms)])
            anchor_pos = pos[anchor_idx]
            support_indices = [i for i in range(len(atoms)) if i not in set(group.tolist())]
            if not support_indices:
                continue
            support_pos = pos[support_indices]
            deltas, _ = get_distances(anchor_pos, support_pos, cell=cell, pbc=pbc)
            dxy = np.linalg.norm(deltas[0, :, :2], axis=1)
            dz = anchor_pos[2] - support_pos[:, 2]

            lateral_mask = dxy < support_xy_tol
            dz_lateral = dz[lateral_mask]
            support_mask = (dz_lateral > 0) & (dz_lateral < z_max_support)
            if not np.any(support_mask):
                return True
        return False

    def _build_site_xy(self) -> np.ndarray:
        if self.top_layer_element is None:
            return np.empty((0, 2), dtype=float)
        if self.site_type == "atop":
            return np.asarray(
                get_toplayer_xy(self.atoms, element=self.top_layer_element), dtype=float
            )

        atop_xy = get_toplayer_xy(self.atoms, element=self.top_layer_element)
        hollow_xy = get_hollow_xy(atop_xy, self.atoms.get_cell().array)
        return np.asarray(
            classify_hollow_sites(
                self.atoms,
                hollow_xy,
                element=self.top_layer_element,
                xy_tol=self.xy_tol,
                stacking=self.site_type,
            ),
            dtype=float,
        )

    def _get_site_xy(self) -> np.ndarray:
        if self._site_xy is None:
            self._site_xy = self._build_site_xy()
        return self._site_xy

    def _propose_displacement(self) -> Optional[Atoms]:
        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        anchor_idx = self.ads_anchor_indices[group_id]
        anchor_pos = self.atoms.positions[anchor_idx].copy()
        relative = self._current_group_relative_positions(group)
        all_pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        xy_matrix = cell[:2, :2]

        for _ in range(self.max_displacement_trials):
            delta = self.rng.normal(0.0, self.displacement_sigma, size=2)
            new_xy = anchor_pos[:2] + delta
            if any(pbc[:2]):
                frac = np.linalg.solve(xy_matrix.T, new_xy)
                frac = frac % 1.0
                new_xy = np.dot(xy_matrix.T, frac)

            new_xyz = np.zeros((1, 3))
            new_xyz[0, :2] = new_xy[:2]
            all_xyz = np.zeros_like(all_pos)
            all_xyz[:, :2] = all_pos[:, :2]
            dxy = get_distances(new_xyz, all_xyz, cell=cell, pbc=pbc)[1].flatten()

            dxy[group] = np.inf
            support_indices = np.where(dxy < self.support_xy_tol)[0]
            if len(support_indices) == 0:
                new_z = float(anchor_pos[2])
            else:
                new_z = float(np.max(all_pos[support_indices, 2]) + self.vertical_offset)

            new_anchor = np.array([new_xy[0], new_xy[1], new_z], dtype=float)
            trial_positions = new_anchor + relative
            if self._group_positions_are_valid(group, trial_positions):
                atoms_new = self.atoms.copy()
                atoms_new.positions[group] = trial_positions
                return atoms_new

        return None

    def _propose_site_hop(self) -> Optional[Atoms]:
        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        site_xy = self._get_site_xy()
        if site_xy.size == 0:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        anchor_idx = self.ads_anchor_indices[group_id]
        current_anchor = self.atoms.positions[anchor_idx].copy()
        relative = self._current_group_relative_positions(group)
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()

        site_order = self.rng.permutation(len(site_xy))
        for site_idx in site_order:
            xy = np.asarray(site_xy[site_idx], dtype=float)
            delta_xyz = np.zeros((1, 3), dtype=float)
            delta_xyz[0, :2] = current_anchor[:2] - xy[:2]
            mic_xy = get_distances(
                np.zeros((1, 3)), delta_xyz, cell=cell, pbc=pbc
            )[1].flatten()[0]
            if mic_xy < 0.5 * self.xy_tol:
                continue

            new_z = self._candidate_support_z(xy, exclude_indices=group)
            if new_z is None:
                continue

            new_anchor = np.array([xy[0], xy[1], new_z], dtype=float)
            trial_positions = new_anchor + relative
            if not self._group_positions_are_valid(group, trial_positions):
                continue

            atoms_new = self.atoms.copy()
            atoms_new.positions[group] = trial_positions
            return atoms_new

        return None

    def _propose_reorientation(self) -> Optional[Atoms]:
        if (not self.is_molecular_adsorbate) or self.rotation_max_angle_rad <= 0.0:
            return None

        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        anchor_idx = self.ads_anchor_indices[group_id]
        anchor_pos = self.atoms.positions[anchor_idx].copy()
        relative = self._current_group_relative_positions(group)

        if np.allclose(relative, 0.0):
            return None

        for _ in range(self.max_reorientation_trials):
            axis = self.rng.normal(size=3)
            angle = self.rng.uniform(
                -self.rotation_max_angle_rad, self.rotation_max_angle_rad
            )
            if np.linalg.norm(axis) <= 1e-12 or abs(angle) <= 1e-12:
                continue

            rotation = _rotation_matrix(axis, angle)
            trial_relative = relative @ rotation.T
            trial_positions = anchor_pos + trial_relative
            if not self._group_positions_are_valid(group, trial_positions):
                continue

            atoms_new = self.atoms.copy()
            atoms_new.positions[group] = trial_positions
            return atoms_new

        return None

    def _propose_move(self) -> Optional[Atoms]:
        if self.move_mode == "displacement":
            return self._propose_displacement()
        if self.move_mode == "site_hop":
            return self._propose_site_hop()
        if self.move_mode == "reorientation":
            return self._propose_reorientation()

        move_selector = self.rng.random()
        if move_selector < self.site_hop_prob:
            return self._propose_site_hop()
        if move_selector < (self.site_hop_prob + self.reorientation_prob):
            trial = self._propose_reorientation()
            if trial is not None:
                return trial
        return self._propose_displacement()

    def _open_optional_traj(self, filename: Optional[str]) -> Optional[Trajectory]:
        if not filename:
            return None
        mode = "a" if os.path.exists(filename) and os.path.getsize(filename) > 0 else "w"
        return Trajectory(filename, mode)

    def run(
        self,
        nsweeps: int,
        traj_file: str,
        interval: int = 10,
        sample_interval: int = 1,
        equilibration: int = 0,
    ) -> Dict[str, float]:
        self.traj_file = traj_file

        mode = "a" if os.path.exists(self.traj_file) and os.path.getsize(self.traj_file) > 0 else "w"
        traj_writer = Trajectory(self.traj_file, mode)
        accepted_writer = self._open_optional_traj(self.accepted_traj_file)
        rejected_writer = self._open_optional_traj(self.rejected_traj_file)
        attempted_writer = self._open_optional_traj(self.attempted_traj_file)

        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.accepted_moves = 0
        self.total_moves = 0

        if len(self.ads_groups) == 0:
            logger.warning("AdsorbateCMC run started with no adsorbates present.")

        for sweep in range(nsweeps):
            beta = 1.0 / (8.617333e-5 * self.T)
            moves_this_sweep = self._moves_per_sweep()

            for i in range(moves_this_sweep):
                self.total_moves += 1
                atoms_trial = self._propose_move()
                if atoms_trial is None:
                    continue

                if attempted_writer is not None:
                    attempted_writer.write(atoms_trial)

                if self.relax:
                    atoms_trial, converged = self.relax_structure(
                        atoms_trial, move_ind=[self.sweep, i]
                    )
                    if not converged:
                        continue
                    if self.has_detached_functional_groups(
                        atoms_trial, detach_tol=self.detach_tol
                    ):
                        if rejected_writer is not None:
                            rejected_writer.write(atoms_trial)
                        continue

                if self.has_afloat_adsorbates(
                    atoms_trial,
                    support_xy_tol=self.support_xy_tol,
                    z_max_support=self.z_max_support,
                ):
                    if rejected_writer is not None:
                        rejected_writer.write(atoms_trial)
                    continue

                e_new = self.get_potential_energy(atoms_trial)
                delta_e = e_new - self.e_old

                if self._metropolis_accept(delta_e, beta=beta):
                    self.atoms = atoms_trial
                    self.atoms.calc = self.calculator
                    self.e_old = e_new
                    self.accepted_moves += 1
                    if self.relax:
                        self._site_xy = None
                    self._update_indices()
                    if accepted_writer is not None:
                        accepted_writer.write(self.atoms)
                elif rejected_writer is not None:
                    rejected_writer.write(atoms_trial)

            self.sweep += 1

            if sweep >= equilibration and (sweep + 1) % sample_interval == 0:
                self.sum_E += self.e_old
                self.sum_E_sq += self.e_old**2
                self.n_samples += 1

            if (sweep + 1) % interval == 0:
                traj_writer.write(self.atoms)
                with open(self.thermo_file, "a") as handle:
                    handle.write(f"{self.sweep} {self.e_old:.6f}\n")

                acc = (
                    (self.accepted_moves / self.total_moves * 100.0)
                    if self.total_moves
                    else 0.0
                )
                avg = self.sum_E / self.n_samples if self.n_samples else self.e_old
                cv = 0.0
                if self.n_samples > 1:
                    var = (self.sum_E_sq / self.n_samples) - (avg**2)
                    cv = var / (8.617333e-5 * self.T**2)

                logger.info(
                    f"T={self.T:4.0f}K | {self.sweep:6d} | "
                    f"E: {self.e_old:10.4f} | Avg: {avg:10.4f} | "
                    f"Cv: {cv:8.4f} | Acc: {acc:4.1f}% | Nads: {len(self.ads_groups):4d}"
                )

            if (
                self.checkpoint_interval > 0
                and self.sweep % self.checkpoint_interval == 0
            ):
                self._save_checkpoint()

        self._save_checkpoint()
        traj_writer.close()
        for writer in (accepted_writer, rejected_writer, attempted_writer):
            if writer is not None:
                writer.close()

        final_avg = self.sum_E / self.n_samples if self.n_samples else self.e_old
        final_cv = 0.0
        if self.n_samples > 1:
            var = (self.sum_E_sq / self.n_samples) - (final_avg**2)
            final_cv = var / (8.617333e-5 * self.T**2)

        return {
            "T": self.T,
            "energy": final_avg,
            "cv": final_cv,
            "acceptance": (
                (self.accepted_moves / self.total_moves * 100.0)
                if self.total_moves
                else 0.0
            ),
            "n_adsorbates": len(self.ads_groups),
        }
