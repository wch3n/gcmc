import logging
import numpy as np
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ase import Atoms
from ase import units
from ase.build import make_supercell
from ase.constraints import FixCartesian
from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.symbols import string2symbols

from .base import SurfaceMCBase
from .constants import ADSORBATE_TAG_OFFSET, KB_EV_PER_K
from .utils import (
    _select_site_layers_for_coverage,
    build_surface_site_registry,
)

logger = logging.getLogger("mc")


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


def _normalize_site_types(
    site_type: Union[str, Sequence[str]],
) -> Tuple[str, ...]:
    if isinstance(site_type, str):
        tokens = tuple(token for token in site_type.replace(",", " ").split() if token)
        normalized = tokens if tokens else (site_type,)
    else:
        normalized = tuple(str(token) for token in site_type)
    normalized = tuple(
        "atop" if token.lower() == "top" else token.lower() for token in normalized
    )
    if "all" in normalized:
        return ("atop", "bridge", "fcc", "hcp")
    allowed = {"atop", "bridge", "fcc", "hcp"}
    if not set(normalized).issubset(allowed):
        raise ValueError(
            "site_type must contain only 'atop', 'bridge', 'fcc', 'hcp', or 'all'."
        )
    return normalized


def _place_adsorbate_template(
    atoms: Atoms,
    adsorbate_template: Atoms,
    *,
    anchor_index: int,
    site_registry: Sequence[Dict[str, object]],
    coverage: float,
    seed: int,
) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms_new = atoms.copy()
    relative = (
        adsorbate_template.get_positions()
        - adsorbate_template.positions[anchor_index].copy()
    )
    candidate_sites = [
        row
        for row in site_registry
        if np.isfinite(float(row["suggested_z_A"]))
        and not bool(row["blocked_by_termination"])
    ]
    n_sites = len(candidate_sites)
    if n_sites == 0:
        raise RuntimeError("*** NO ELIGIBLE REGISTRY SITES AVAILABLE ***")

    tags = np.asarray(atoms_new.get_tags(), dtype=int)
    if len(tags) != len(atoms_new):
        tags = np.zeros(len(atoms_new), dtype=int)

    group_id = 0
    for layer_indices in _select_site_layers_for_coverage(n_sites, coverage, rng):
        for site_idx in np.asarray(layer_indices, dtype=int):
            site = candidate_sites[int(site_idx)]
            anchor_pos = np.array(
                [
                    float(site["xy"][0]),
                    float(site["xy"][1]),
                    float(site["suggested_z_A"]),
                ],
                dtype=float,
            )

            group_tag = ADSORBATE_TAG_OFFSET + group_id
            for symbol, rel in zip(adsorbate_template.get_chemical_symbols(), relative):
                atoms_new.append(symbol)
                atoms_new.positions[-1] = anchor_pos + rel
                tags = np.append(tags, group_tag)
            group_id += 1

    atoms_new.set_tags(tags)
    return atoms_new


class AdsorbateCMC(SurfaceMCBase):
    """
    Canonical Monte Carlo for fixed-loading adsorbates on a surface.

    This class is designed to match the `AlloyCMC`/`ReplicaExchange` worker
    interface. Molecular adsorbates are handled as rigid groups whose anchor
    atom defines support and site-hop placement.

    Tolerance conventions:
    - ``min_clearance`` is the minimum full 3D adsorbate/slab clearance.
    - ``site_match_tol`` controls how strictly high-symmetry sites are matched
      on the instantaneous surface.
    - ``surface_layer_tol`` controls z-layer clustering for the exposed surface.
    - ``termination_clearance`` blocks sites or trial placements that come too
      close to surface terminations.
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
        site_elements: Optional[Union[str, Sequence[str]]] = None,
        surface_side: str = "top",
        coverage: Optional[float] = None,
        repeat: Sequence[int] = (1, 1, 1),
        supercell_matrix: Optional[Sequence[Sequence[int]]] = None,
        site_type: Union[str, Sequence[str]] = "fcc",
        move_mode: str = "displacement",
        site_hop_prob: float = 0.5,
        reorientation_prob: float = 0.2,
        displacement_sigma: float = 1.5,
        max_displacement_trials: int = 10,
        max_reorientation_trials: Optional[int] = None,
        rotation_max_angle_deg: float = 25.0,
        min_clearance: float = 0.8,
        site_match_tol: float = 0.6,
        surface_layer_tol: float = 0.5,
        termination_clearance: float = 0.75,
        bridge_cutoff: Optional[float] = None,
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
        allow_ambiguous_empty_adsorbates: bool = False,
        enable_hybrid_md: bool = False,
        md_move_prob: float = 0.1,
        md_steps: int = 50,
        md_timestep_fs: float = 1.0,
        md_ensemble: str = "nve",
        md_accept_mode: str = "potential",
        md_friction: float = 0.01,
        md_planar: bool = False,
        md_planar_axis: int = 2,
        md_init_momenta: bool = True,
        md_remove_drift: bool = True,
        **kwargs,
    ):
        if isinstance(atoms, str):
            self.atoms = read(atoms)
        else:
            self.atoms = atoms.copy()

        if supercell_matrix is not None:
            self.atoms = make_supercell(
                self.atoms, np.asarray(supercell_matrix, dtype=int)
            )

        repeat = tuple(int(value) for value in repeat)
        if len(repeat) != 3:
            raise ValueError("repeat must be a 3-element sequence.")
        if repeat != (1, 1, 1):
            self.atoms = self.atoms.repeat(repeat)

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
        self.allow_ambiguous_empty_adsorbates = bool(
            allow_ambiguous_empty_adsorbates
        )
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

        support_xy_tol = max(1.2, 2.5 * float(site_match_tol))
        same_site_tol = max(0.2, 0.5 * float(site_match_tol))

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
        if min_clearance <= 0.0:
            raise ValueError("min_clearance must be > 0.")
        if site_match_tol <= 0.0:
            raise ValueError("site_match_tol must be > 0.")
        if surface_layer_tol <= 0.0:
            raise ValueError("surface_layer_tol must be > 0.")
        if termination_clearance < 0.0:
            raise ValueError("termination_clearance must be >= 0.")
        if bridge_cutoff is not None and bridge_cutoff <= 0.0:
            raise ValueError("bridge_cutoff must be > 0 when provided.")

        self.T = T
        if surface_side not in {"top", "bottom"}:
            raise ValueError("surface_side must be 'top' or 'bottom'.")
        self.top_layer_element = (
            top_layer_element
            if top_layer_element is not None
            else (substrate_elements[0] if substrate_elements else None)
        )
        if site_elements is None:
            resolved_site_elements: Tuple[str, ...] = (
                (self.top_layer_element,) if self.top_layer_element is not None else ()
            )
        elif isinstance(site_elements, str):
            tokens = tuple(
                token for token in site_elements.replace(",", " ").split() if token
            )
            resolved_site_elements = tokens if tokens else tuple(string2symbols(site_elements))
        else:
            resolved_site_elements = tuple(str(el) for el in site_elements)
        self.site_elements = resolved_site_elements
        if not self.site_elements:
            raise ValueError(
                "site_elements resolved to an empty set. Provide site_elements or top_layer_element."
            )
        self.surface_side = surface_side
        self.coverage = coverage
        self.site_types = _normalize_site_types(site_type)
        self.site_type = self.site_types[0] if len(self.site_types) == 1 else "all"
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
        self.min_clearance = float(min_clearance)
        self.site_match_tol = float(site_match_tol)
        self.same_site_tol = float(same_site_tol)
        self.support_xy_tol = float(support_xy_tol)
        self.z_max_support = z_max_support
        self.vertical_offset = vertical_offset
        self.surface_layer_tol = float(surface_layer_tol)
        self.bridge_cutoff = bridge_cutoff
        self.bridge_cutoff_scale = 1.15
        self.termination_clearance = float(termination_clearance)
        self.relax = relax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.rejected_traj_file = rejected_traj_file
        self.attempted_traj_file = attempted_traj_file
        self.thermo_file = thermo_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self._site_registry: Optional[list[dict[str, object]]] = None
        self.enable_hybrid_md = bool(enable_hybrid_md)
        self.md_move_prob = float(md_move_prob)
        self.md_steps = int(md_steps)
        self.md_timestep_fs = float(md_timestep_fs)
        self.md_ensemble = str(md_ensemble).lower()
        self.md_accept_mode = str(md_accept_mode).lower()
        self.md_friction = float(md_friction)
        self.md_planar = bool(md_planar)
        self.md_planar_axis = int(md_planar_axis)
        self.md_init_momenta = bool(md_init_momenta)
        self.md_remove_drift = bool(md_remove_drift)
        if not (0.0 <= self.md_move_prob <= 1.0):
            raise ValueError("md_move_prob must be in [0, 1].")
        if self.md_steps < 1:
            raise ValueError("md_steps must be >= 1.")
        if self.md_timestep_fs <= 0.0:
            raise ValueError("md_timestep_fs must be > 0.")
        if self.md_ensemble not in ("nve", "langevin"):
            raise ValueError("md_ensemble must be 'nve' or 'langevin'.")
        if self.md_accept_mode not in ("potential", "hamiltonian"):
            raise ValueError("md_accept_mode must be 'potential' or 'hamiltonian'.")
        if self.md_planar_axis not in (0, 1, 2):
            raise ValueError("md_planar_axis must be 0, 1, or 2.")
        if self.md_accept_mode == "hamiltonian":
            if self.md_ensemble != "nve":
                raise ValueError(
                    "md_accept_mode='hamiltonian' requires md_ensemble='nve'."
                )
            if not self.md_init_momenta:
                raise ValueError(
                    "md_accept_mode='hamiltonian' requires md_init_momenta=True."
                )

        self._update_indices()
        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.accepted_moves = 0
        self.total_moves = 0
        self.md_attempted_moves = 0
        self.md_accepted_moves = 0
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

        if self.allow_ambiguous_empty_adsorbates:
            return []

        raise ValueError(
            "Could not infer adsorbate groups automatically. For molecular adsorbates "
            "with species overlapping the slab/functionals, use a tagged structure, "
            "AdsorbateCMC.from_clean_surface(...), or allow_ambiguous_empty_adsorbates=True "
            "when starting from a known clean surface."
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

    def _refresh_cached_state(self) -> None:
        self._site_registry = None
        self._update_indices()

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
        site_elements: Optional[Union[str, Sequence[str]]] = None,
        surface_side: str = "top",
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: Union[str, Sequence[str]] = "fcc",
        min_clearance: float = 0.8,
        site_match_tol: float = 0.6,
        surface_layer_tol: float = 0.5,
        termination_clearance: float = 0.75,
        bridge_cutoff: Optional[float] = None,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        seed: int = 81,
        initial_traj_file: str = "adsorbate_cmc_initial.traj",
        **kwargs,
    ) -> "AdsorbateCMC":
        support_xy_tol = max(1.2, 2.5 * float(site_match_tol))
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
        site_registry = build_surface_site_registry(
            atoms,
            site_elements=(
                site_elements
                if site_elements is not None
                else ((top_layer_element,) if top_layer_element is not None else ())
            ),
            substrate_elements=substrate_elements,
            surface_side=surface_side,
            site_types=site_type,
            layer_tol=surface_layer_tol,
            xy_tol=site_match_tol,
            bridge_cutoff=bridge_cutoff,
            bridge_cutoff_scale=1.15,
            support_xy_tol=support_xy_tol,
            vertical_offset=vertical_offset,
            termination_elements=functional_elements,
            min_termination_dist=termination_clearance,
        )
        atoms_with_ads = _place_adsorbate_template(
            atoms,
            adsorbate_template,
            anchor_index=int(adsorbate_anchor_index),
            site_registry=site_registry,
            coverage=coverage,
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
            site_elements=site_elements,
            surface_side=surface_side,
            coverage=coverage,
            site_type=site_type,
            min_clearance=min_clearance,
            site_match_tol=site_match_tol,
            surface_layer_tol=surface_layer_tol,
            termination_clearance=termination_clearance,
            vertical_offset=vertical_offset,
            bridge_cutoff=bridge_cutoff,
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
        self._site_registry = None
        self._update_indices()
        logger.info(f"[{self.T:.0f}K] Resumed adsorbate MC from checkpoint.")

    def _metropolis_accept(self, delta_e: float, beta: Optional[float] = None) -> bool:
        if delta_e < 0.0:
            return True
        if beta is None:
            beta = 1.0 / (KB_EV_PER_K * self.T)
        return self.rng.random() < np.exp(-delta_e * beta)

    def _apply_planar_constraint(self, atoms_obj: Atoms) -> None:
        if not self.md_planar:
            return

        mask = [False, False, False]
        mask[self.md_planar_axis] = True
        planar_fix = FixCartesian(np.arange(len(atoms_obj)), mask=mask)

        existing = atoms_obj.constraints
        if existing is None:
            atoms_obj.set_constraint(planar_fix)
            return
        if isinstance(existing, (list, tuple)):
            atoms_obj.set_constraint(list(existing) + [planar_fix])
            return
        atoms_obj.set_constraint([existing, planar_fix])

    def _project_momenta_to_plane(self, atoms_obj: Atoms) -> None:
        if not self.md_planar:
            return

        momenta = atoms_obj.get_momenta()
        momenta[:, self.md_planar_axis] = 0.0
        atoms_obj.set_momenta(momenta)

    def _propose_md_move(self) -> Tuple[Optional[Atoms], float, float]:
        atoms_trial = self.atoms.copy()
        atoms_trial.calc = self.calculator
        self._apply_planar_constraint(atoms_trial)

        if self.md_init_momenta:
            MaxwellBoltzmannDistribution(
                atoms_trial,
                temperature_K=self.T,
                rng=self.rng,
            )
            if self.md_remove_drift:
                Stationary(atoms_trial)
            self._project_momenta_to_plane(atoms_trial)

        e_old = self.e_old
        k_old = atoms_trial.get_kinetic_energy()
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
            logger.warning(f"Adsorbate MD trial move failed: {exc}")
            return None, 0.0, 0.0

        delta_e = e_new - e_old
        k_new = atoms_trial.get_kinetic_energy()
        delta_h = (e_new + k_new) - (e_old + k_old)
        return atoms_trial, delta_e, delta_h

    def _moves_per_sweep(self) -> int:
        n_ads = len(self.ads_groups)
        if n_ads == 0:
            return 0
        if self.move_mode in ("site_hop", "hybrid"):
            active_sites = sum(
                1
                for row in self._get_site_registry()
                if np.isfinite(float(row.get("suggested_z_A", np.nan)))
                and not bool(row.get("blocked_by_termination", False))
            )
            if active_sites > 0:
                return int(active_sites)
        return int(n_ads)

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
        if self.surface_side == "top":
            anchor_z = np.max(all_pos[support_indices, 2]) + self.vertical_offset
        else:
            anchor_z = np.min(all_pos[support_indices, 2]) - self.vertical_offset
        return float(anchor_z)

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
        return float(np.min(dists)) >= self.min_clearance

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
            side_sign = 1.0 if self.surface_side == "top" else -1.0
            mask = (dxy < support_xy_tol) & ((side_sign * dz) > z_tol)
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

        ads_mask = np.zeros(len(atoms), dtype=bool)
        anchor_indices = []
        for group in groups:
            group_arr = np.asarray(group, dtype=int)
            ads_mask[group_arr] = True
            anchor_indices.append(
                int(group_arr[self._group_anchor_local_index(group_arr, atoms=atoms)])
            )

        support_indices = np.where(~ads_mask)[0]
        if support_indices.size == 0:
            return True

        anchor_pos = pos[np.asarray(anchor_indices, dtype=int)]
        support_pos = pos[support_indices]
        deltas = get_distances(anchor_pos, support_pos, cell=cell, pbc=pbc)[0]
        dxy = np.linalg.norm(deltas[:, :, :2], axis=2)
        dz = anchor_pos[:, None, 2] - support_pos[None, :, 2]

        side_sign = 1.0 if self.surface_side == "top" else -1.0
        support_mask = (
            (dxy < support_xy_tol)
            & ((side_sign * dz) > 0)
            & ((side_sign * dz) < z_max_support)
        )
        return bool(np.any(np.sum(support_mask, axis=1) == 0))

    def _build_site_registry(self) -> list[dict[str, object]]:
        return build_surface_site_registry(
            self.atoms,
            site_elements=self.site_elements,
            substrate_elements=self.substrate_elements,
            surface_side=self.surface_side,
            site_types=self.site_types,
            layer_tol=self.surface_layer_tol,
            xy_tol=self.site_match_tol,
            bridge_cutoff=self.bridge_cutoff,
            bridge_cutoff_scale=self.bridge_cutoff_scale,
            support_xy_tol=self.support_xy_tol,
            vertical_offset=self.vertical_offset,
            termination_elements=self.functional_elements,
            min_termination_dist=self.termination_clearance,
        )

    def _get_site_registry(self) -> list[dict[str, object]]:
        if self._site_registry is None:
            self._site_registry = self._build_site_registry()
        return self._site_registry

    def _group_clears_terminations(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if self.termination_clearance <= 0.0:
            return True
        if atoms is None:
            atoms = self.atoms
        if not self.functional_elements:
            return True

        term_indices = np.asarray(self.func_indices, dtype=int)
        if term_indices.size == 0:
            return True
        group = np.asarray(group, dtype=int)
        if group.size > 0:
            term_indices = term_indices[~np.isin(term_indices, group)]
            if term_indices.size == 0:
                return True

        dists = get_distances(
            np.asarray(trial_positions, dtype=float),
            atoms.get_positions()[term_indices],
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )[1]
        return float(np.min(dists)) >= self.termination_clearance

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
            if not self._group_positions_are_valid(group, trial_positions):
                continue
            if not self._group_clears_terminations(group, trial_positions):
                continue
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

        site_registry = self._get_site_registry()
        if not site_registry:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        anchor_idx = self.ads_anchor_indices[group_id]
        current_anchor = self.atoms.positions[anchor_idx].copy()
        relative = self._current_group_relative_positions(group)
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()

        site_order = self.rng.permutation(len(site_registry))
        for site_idx in site_order:
            site = site_registry[int(site_idx)]
            if bool(site.get("blocked_by_termination", False)):
                continue
            suggested_z = float(site.get("suggested_z_A", np.nan))
            if not np.isfinite(suggested_z):
                continue
            xy = np.asarray(site["xy"], dtype=float)
            delta_xyz = np.zeros((1, 3), dtype=float)
            delta_xyz[0, :2] = current_anchor[:2] - xy[:2]
            mic_xy = get_distances(
                np.zeros((1, 3)), delta_xyz, cell=cell, pbc=pbc
            )[1].flatten()[0]
            if mic_xy < self.same_site_tol:
                continue

            new_anchor = np.array([xy[0], xy[1], suggested_z], dtype=float)
            trial_positions = new_anchor + relative
            if not self._group_positions_are_valid(group, trial_positions):
                continue
            if not self._group_clears_terminations(group, trial_positions):
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
            if not self._group_clears_terminations(group, trial_positions):
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
        self.md_attempted_moves = 0
        self.md_accepted_moves = 0

        if len(self.ads_groups) == 0:
            logger.warning("AdsorbateCMC run started with no adsorbates present.")

        for sweep in range(nsweeps):
            beta = 1.0 / (KB_EV_PER_K * self.T)
            moves_this_sweep = self._moves_per_sweep()

            for i in range(moves_this_sweep):
                self.total_moves += 1
                do_md = self.enable_hybrid_md and self.rng.random() < self.md_move_prob
                if do_md:
                    self.md_attempted_moves += 1
                    atoms_trial, delta_e, delta_h = self._propose_md_move()
                    if atoms_trial is None:
                        continue

                    if attempted_writer is not None:
                        attempted_writer.write(atoms_trial)

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

                    md_delta = (
                        delta_h if self.md_accept_mode == "hamiltonian" else delta_e
                    )
                    if self._metropolis_accept(md_delta, beta=beta):
                        self.e_old += delta_e
                        self.accepted_moves += 1
                        self.md_accepted_moves += 1
                        self.atoms.positions = atoms_trial.positions
                        self.atoms.cell = atoms_trial.cell
                        self._site_registry = None
                        if accepted_writer is not None:
                            accepted_writer.write(self.atoms)
                    elif rejected_writer is not None:
                        rejected_writer.write(atoms_trial)
                    continue

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
                        self._site_registry = None
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
                    cv = var / (KB_EV_PER_K * self.T**2)

                logger.info(
                    f"T={self.T:4.0f}K | {self.sweep:6d} | "
                    f"E: {self.e_old:10.4f} | Avg: {avg:10.4f} | "
                    f"Cv: {cv:8.4f} | Acc: {acc:4.1f}% | Nads: {len(self.ads_groups):4d}"
                    + (
                        ""
                        if not self.enable_hybrid_md
                        else (
                            f" | MD: {self.md_accepted_moves}/{self.md_attempted_moves}"
                            f" ({((self.md_accepted_moves / self.md_attempted_moves) * 100.0) if self.md_attempted_moves else 0.0:4.1f}%)"
                            f" | MD_frac: {((self.md_attempted_moves / self.total_moves) * 100.0) if self.total_moves else 0.0:4.1f}%"
                            f" | MD_accept: {self.md_accept_mode}"
                            f" | planar: {self.md_planar}"
                        )
                    )
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
            final_cv = var / (KB_EV_PER_K * self.T**2)

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
            "md_attempted": self.md_attempted_moves,
            "md_accepted": self.md_accepted_moves,
            "md_accept_mode": self.md_accept_mode,
        }
