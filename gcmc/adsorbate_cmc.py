import logging
import numpy as np
import os
import pickle
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ase import Atoms
from ase import units
from ase.build import make_supercell
from ase.constraints import FixCartesian
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.symbols import string2symbols

from .base import SurfaceMCBase
from .constants import ADSORBATE_TAG_OFFSET, KB_EV_PER_K
from .adsorbate_move_proposals import AdsorbateMoveProposalMixin, MoveProposal
from .utils import (
    _select_site_layers_for_coverage,
    build_surface_site_registry,
)

logger = logging.getLogger("mc")


def _normalize_puckering_heights(
    config: Optional[Mapping[str, Mapping[str, object]]],
    *,
    eligible_elements: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("puckering_heights must be a mapping keyed by element.")

    eligible = set(str(element) for element in eligible_elements)
    normalized: Dict[str, Tuple[float, float]] = {}
    for raw_element, raw_parameters in config.items():
        element = str(raw_element)
        if element not in eligible:
            raise ValueError(
                f"puckering_heights contains {element!r}, which is not listed in "
                "puckering_elements."
            )
        if not isinstance(raw_parameters, Mapping):
            raise ValueError(
                f"puckering_heights[{element!r}] must be a mapping with height_A."
            )
        if "height_A" not in raw_parameters:
            raise ValueError(
                f"puckering_heights[{element!r}] must define height_A."
            )
        height = float(raw_parameters["height_A"])
        if height < 0.0:
            raise ValueError(
                f"puckering_heights[{element!r}].height_A must be >= 0."
            )
        raw_jitter = raw_parameters.get("height_jitter_A")
        jitter = 0.1 * height if raw_jitter is None else float(raw_jitter)
        if jitter < 0.0:
            raise ValueError(
                f"puckering_heights[{element!r}].height_jitter_A must be >= 0."
            )
        normalized[element] = (height, jitter)
    return normalized


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
            try:
                return read(adsorbate)
            except Exception:
                return read(adsorbate, format="vasp")
        key = str(adsorbate).upper()
        if key == "OH":
            return Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)])
        if key == "OOH":
            return Atoms(
                "OOH",
                positions=[
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.46),
                    (0.79, 0.0, 2.01),
                ],
            )
        symbols = string2symbols(adsorbate)
        if len(symbols) != 1:
            raise ValueError(
                "String adsorbates must be a single chemical symbol or a structure file. "
                "For molecular adsorbates, pass an ASE Atoms template or a file path."
            )
        return Atoms(symbols=symbols, positions=[(0.0, 0.0, 0.0)])
    raise TypeError("adsorbate must be None, a chemical symbol/path string, or ASE Atoms.")


def _normalize_anchor_mode(mode: object) -> str:
    token = str(mode or "atom").strip().lower().replace("-", "_")
    aliases = {
        "index": "atom",
        "atom_index": "atom",
        "com": "center_of_mass",
        "center": "center_of_mass",
        "center_of_geometry": "centroid",
        "midpoint": "centroid",
    }
    token = aliases.get(token, token)
    if token not in {"atom", "center_of_mass", "centroid"}:
        raise ValueError(
            "adsorbate anchor mode must be 'atom', 'center_of_mass', or 'centroid'."
        )
    return token


def _normalize_anchor_reference_indices(
    *,
    mode: str,
    anchor_index: int,
    atom_indices: Optional[Sequence[int]],
    size: int,
) -> tuple[int, ...]:
    if mode == "atom":
        indices = (int(anchor_index),)
    elif atom_indices is None:
        indices = tuple(range(size))
    else:
        indices = tuple(int(idx) for idx in atom_indices)
    if not indices:
        raise ValueError("adsorbate anchor atom_indices must not be empty.")
    for idx in indices:
        if not (0 <= idx < int(size)):
            raise ValueError("adsorbate anchor atom_indices entries are out of range.")
    return indices


def _anchor_reference_position(
    template: Atoms,
    *,
    mode: str,
    anchor_index: int,
    atom_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    positions = np.asarray(template.get_positions(), dtype=float)
    mode = _normalize_anchor_mode(mode)
    indices = _normalize_anchor_reference_indices(
        mode=mode,
        anchor_index=int(anchor_index),
        atom_indices=atom_indices,
        size=len(template),
    )
    if mode == "atom":
        return positions[int(anchor_index)].copy()
    selected = positions[np.asarray(indices, dtype=int)]
    if mode == "centroid":
        return np.mean(selected, axis=0)
    masses = np.asarray(
        [atomic_masses[atomic_numbers[template[int(idx)].symbol]] for idx in indices],
        dtype=float,
    )
    if not np.all(np.isfinite(masses)) or float(np.sum(masses)) <= 0.0:
        return np.mean(selected, axis=0)
    return np.average(selected, axis=0, weights=masses)


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
    anchor_mode: str = "atom",
    anchor_atom_indices: Optional[Sequence[int]] = None,
    site_registry: Sequence[Dict[str, object]],
    coverage: float,
    seed: int,
) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms_new = atoms.copy()
    relative = (
        adsorbate_template.get_positions()
        - _anchor_reference_position(
            adsorbate_template,
            mode=anchor_mode,
            anchor_index=int(anchor_index),
            atom_indices=anchor_atom_indices,
        )
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

    def _anchor_position_from_site(site: Dict[str, object]) -> np.ndarray:
        support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
        support_indices = support_indices[
            (support_indices >= 0) & (support_indices < len(atoms_new))
        ]
        if support_indices.size > 0:
            side = str(site.get("surface_side", "top")).lower()
            if side == "bottom":
                support_z = float(np.min(atoms_new.positions[support_indices, 2]))
            else:
                support_z = float(np.max(atoms_new.positions[support_indices, 2]))
            suggested_z = support_z + float(site["suggested_z_A"] - site["anchor_z_A"])
        else:
            suggested_z = float(site["suggested_z_A"])
        return np.array(
            [
                float(site["xy"][0]),
                float(site["xy"][1]),
                suggested_z,
            ],
            dtype=float,
        )

    group_id = 0
    for layer_indices in _select_site_layers_for_coverage(n_sites, coverage, rng):
        for site_idx in np.asarray(layer_indices, dtype=int):
            site = candidate_sites[int(site_idx)]
            anchor_pos = _anchor_position_from_site(site)

            group_tag = ADSORBATE_TAG_OFFSET + group_id
            for symbol, rel in zip(adsorbate_template.get_chemical_symbols(), relative):
                atoms_new.append(symbol)
                atoms_new.positions[-1] = anchor_pos + rel
                tags = np.append(tags, group_tag)
            group_id += 1

    atoms_new.set_tags(tags)
    return atoms_new


class AdsorbateCMC(AdsorbateMoveProposalMixin, SurfaceMCBase):
    """
    Canonical Monte Carlo for fixed-loading adsorbates on a surface.

    This class is designed to match the `AlloyCMC`/`ReplicaExchange` worker
    interface. Molecular adsorbates are handled as rigid groups whose anchor
    atom defines support and site-hop placement.

    Tolerance conventions:
    - ``min_clearance`` is the minimum full 3D adsorbate/slab clearance.
    - ``site_match_tol`` controls how strictly high-symmetry sites are matched
      when the fixed site registry is built from the initial surface.
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
        adsorbate_anchor_mode: str = "atom",
        adsorbate_anchor_atom_indices: Optional[Sequence[int]] = None,
        adsorbate_template_library: Optional[Sequence[Dict[str, object]]] = None,
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
        hop_reorientation_prob: float = 0.0,
        hop_puckering_prob: float = 0.0,
        hop_puckering_reorientation_prob: float = 0.0,
        puckering_prob: float = 0.0,
        puckering_elements: Optional[Union[str, Sequence[str]]] = None,
        puckering_height_A: float = 0.15,
        puckering_height_jitter_A: Optional[float] = None,
        puckering_heights: Optional[Mapping[str, Mapping[str, object]]] = None,
        displacement_sigma: float = 1.5,
        max_displacement_trials: int = 10,
        max_reorientation_trials: Optional[int] = None,
        max_hop_reorientation_trials: Optional[int] = None,
        max_puckering_trials: Optional[int] = None,
        rotation_max_angle_deg: float = 25.0,
        hop_reorientation_angle_deg: float = 180.0,
        min_clearance: float = 0.8,
        adsorbate_surface_clearance_A: float = 0.0,
        adsorbate_surface_xy_tol_A: Optional[float] = None,
        site_match_tol: float = 0.6,
        support_xy_tol: Optional[float] = None,
        termination_site_xy_tol: Optional[float] = None,
        surface_layer_tol: float = 0.5,
        termination_clearance: float = 0.75,
        bridge_cutoff: Optional[float] = None,
        z_max_support: float = 3.5,
        vertical_offset: float = 1.8,
        vertical_adjust_step: float = 0.25,
        max_vertical_adjust: float = 1.5,
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
        debug_traj_interval: int = 1,
        diagnostics_enabled: bool = False,
        diagnostics_log: bool = False,
        diagnostics_top_n: int = 3,
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
        enforce_molecular_integrity: bool = True,
        molecular_bond_stretch_factor: float = 1.35,
        molecular_bond_abs_tol: float = 0.35,
        molecular_upright_atom_indices: Optional[Sequence[int]] = None,
        molecular_upright_min_z_A: Optional[float] = None,
        site_region_center_A: Optional[Sequence[float]] = None,
        site_region_radius_A: Optional[float] = None,
        site_region_distance_metric: str = "xy",
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
        self.adsorbate_anchor_mode = _normalize_anchor_mode(adsorbate_anchor_mode)
        self.adsorbate_anchor_reference_indices = _normalize_anchor_reference_indices(
            mode=self.adsorbate_anchor_mode,
            anchor_index=self.adsorbate_anchor_index,
            atom_indices=adsorbate_anchor_atom_indices,
            size=len(self.adsorbate_template),
        )
        self.adsorbate_size = len(self.adsorbate_template)
        self.is_molecular_adsorbate = self.adsorbate_size > 1
        self.adsorbate_anchor_symbol = self.adsorbate_template[
            self.adsorbate_anchor_index
        ].symbol
        self.adsorbate_symbols = tuple(self.adsorbate_template.get_chemical_symbols())
        self._adsorbate_symbol_signature = tuple(sorted(self.adsorbate_symbols))
        self.adsorbate_template_library = self._normalize_template_library(
            adsorbate_template_library
        )
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

        if support_xy_tol is None:
            support_xy_tol = max(1.2, 2.5 * float(site_match_tol))
        if termination_site_xy_tol is None:
            termination_site_xy_tol = float(support_xy_tol)
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
        if move_mode not in (
            "displacement",
            "channel_hop",
            "site_hop",
            "reorientation",
            "hop_reorientation",
            "hop_puckering",
            "hop_puckering_reorientation",
            "puckering",
            "hybrid",
        ):
            raise ValueError(
                "move_mode must be 'displacement', 'site_hop', 'reorientation', "
                "'channel_hop', 'hop_reorientation', 'hop_puckering', "
                "'hop_puckering_reorientation', 'puckering', or 'hybrid'."
            )
        if not (0.0 <= site_hop_prob <= 1.0):
            raise ValueError("site_hop_prob must be in [0, 1].")
        if not (0.0 <= reorientation_prob <= 1.0):
            raise ValueError("reorientation_prob must be in [0, 1].")
        if not (0.0 <= hop_reorientation_prob <= 1.0):
            raise ValueError("hop_reorientation_prob must be in [0, 1].")
        if not (0.0 <= hop_puckering_prob <= 1.0):
            raise ValueError("hop_puckering_prob must be in [0, 1].")
        if not (0.0 <= hop_puckering_reorientation_prob <= 1.0):
            raise ValueError("hop_puckering_reorientation_prob must be in [0, 1].")
        if not (0.0 <= puckering_prob <= 1.0):
            raise ValueError("puckering_prob must be in [0, 1].")
        if move_mode == "hybrid" and (
            site_hop_prob
            + reorientation_prob
            + hop_reorientation_prob
            + hop_puckering_prob
            + hop_puckering_reorientation_prob
            + puckering_prob
        ) > 1.0:
            raise ValueError(
                "For move_mode='hybrid', site_hop_prob + reorientation_prob "
                "+ hop_reorientation_prob + hop_puckering_prob "
                "+ hop_puckering_reorientation_prob + puckering_prob must be <= 1."
            )
        if puckering_height_A < 0.0:
            raise ValueError("puckering_height_A must be >= 0.")
        if puckering_height_jitter_A is not None and puckering_height_jitter_A < 0.0:
            raise ValueError("puckering_height_jitter_A must be >= 0.")
        if rotation_max_angle_deg < 0.0:
            raise ValueError("rotation_max_angle_deg must be >= 0.")
        if hop_reorientation_angle_deg < 0.0:
            raise ValueError("hop_reorientation_angle_deg must be >= 0.")
        if min_clearance <= 0.0:
            raise ValueError("min_clearance must be > 0.")
        if adsorbate_surface_clearance_A < 0.0:
            raise ValueError("adsorbate_surface_clearance_A must be >= 0.")
        if (
            adsorbate_surface_xy_tol_A is not None
            and adsorbate_surface_xy_tol_A <= 0.0
        ):
            raise ValueError("adsorbate_surface_xy_tol_A must be > 0 when provided.")
        if site_match_tol <= 0.0:
            raise ValueError("site_match_tol must be > 0.")
        if support_xy_tol <= 0.0:
            raise ValueError("support_xy_tol must be > 0.")
        if termination_site_xy_tol <= 0.0:
            raise ValueError("termination_site_xy_tol must be > 0.")
        if surface_layer_tol <= 0.0:
            raise ValueError("surface_layer_tol must be > 0.")
        if termination_clearance < 0.0:
            raise ValueError("termination_clearance must be >= 0.")
        if vertical_adjust_step <= 0.0:
            raise ValueError("vertical_adjust_step must be > 0.")
        if max_vertical_adjust < 0.0:
            raise ValueError("max_vertical_adjust must be >= 0.")
        if bridge_cutoff is not None and bridge_cutoff <= 0.0:
            raise ValueError("bridge_cutoff must be > 0 when provided.")
        if site_region_radius_A is not None and float(site_region_radius_A) <= 0.0:
            raise ValueError("site_region_radius_A must be > 0 when provided.")
        site_region_distance_metric = str(site_region_distance_metric).lower()
        if site_region_distance_metric not in {"xy", "xyz"}:
            raise ValueError("site_region_distance_metric must be 'xy' or 'xyz'.")
        if site_region_center_A is not None:
            center = np.asarray(site_region_center_A, dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError("site_region_center_A must be a finite 3-vector.")
        else:
            center = None

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
        if puckering_elements is None:
            resolved_puckering_elements = resolved_site_elements
        elif isinstance(puckering_elements, str):
            tokens = tuple(
                token
                for token in puckering_elements.replace(",", " ").split()
                if token
            )
            resolved_puckering_elements = (
                tokens if tokens else tuple(string2symbols(puckering_elements))
            )
        else:
            resolved_puckering_elements = tuple(str(el) for el in puckering_elements)
        if not resolved_puckering_elements:
            raise ValueError("puckering_elements resolved to an empty set.")
        self.puckering_elements = resolved_puckering_elements
        self.surface_side = surface_side
        self.coverage = coverage
        self.site_types = _normalize_site_types(site_type)
        self.site_type = self.site_types[0] if len(self.site_types) == 1 else "all"
        self.move_mode = move_mode
        self.site_hop_prob = float(site_hop_prob)
        self.reorientation_prob = float(reorientation_prob)
        self.hop_reorientation_prob = float(hop_reorientation_prob)
        self.hop_puckering_prob = float(hop_puckering_prob)
        self.hop_puckering_reorientation_prob = float(
            hop_puckering_reorientation_prob
        )
        self.puckering_prob = float(puckering_prob)
        spatial_channel_weight = (
            self.site_hop_prob
            + self.hop_reorientation_prob
            + self.hop_puckering_prob
            + self.hop_puckering_reorientation_prob
        )
        self.channel_hop_prob = spatial_channel_weight + self.puckering_prob
        self.channel_reorientation_prob = (
            (
                self.hop_reorientation_prob
                + self.hop_puckering_reorientation_prob
            )
            / spatial_channel_weight
            if spatial_channel_weight > 1e-15
            else 0.0
        )
        self.channel_puckering_enabled = bool(
            self.puckering_prob > 0.0
            or self.hop_puckering_prob > 0.0
            or self.hop_puckering_reorientation_prob > 0.0
            or self.move_mode
            in {
                "channel_hop",
                "hop_puckering",
                "hop_puckering_reorientation",
                "puckering",
            }
        )
        self.puckering_height_A = float(puckering_height_A)
        self.puckering_height_jitter_A = (
            0.1 * self.puckering_height_A
            if puckering_height_jitter_A is None
            else float(puckering_height_jitter_A)
        )
        self.puckering_heights = _normalize_puckering_heights(
            puckering_heights,
            eligible_elements=self.puckering_elements,
        )
        self.displacement_sigma = displacement_sigma
        self.max_displacement_trials = int(max_displacement_trials)
        self.max_reorientation_trials = (
            int(max_reorientation_trials)
            if max_reorientation_trials is not None
            else int(max_displacement_trials)
        )
        self.max_hop_reorientation_trials = (
            int(max_hop_reorientation_trials)
            if max_hop_reorientation_trials is not None
            else int(max_displacement_trials)
        )
        self.max_puckering_trials = (
            int(max_puckering_trials)
            if max_puckering_trials is not None
            else int(max_displacement_trials)
        )
        self._puckering_reference_positions = self.atoms.get_positions().copy()
        self._puckering_surface_support_indices_cache: Optional[np.ndarray] = None
        self._puckering_local_reference_cache: dict[
            tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
        self.rotation_max_angle_rad = np.deg2rad(float(rotation_max_angle_deg))
        self.hop_reorientation_angle_rad = np.deg2rad(
            float(hop_reorientation_angle_deg)
        )
        self.min_clearance = float(min_clearance)
        self.adsorbate_surface_clearance_A = float(adsorbate_surface_clearance_A)
        self.site_match_tol = float(site_match_tol)
        self.same_site_tol = float(same_site_tol)
        self.support_xy_tol = float(support_xy_tol)
        self.adsorbate_surface_xy_tol_A = (
            self.support_xy_tol
            if adsorbate_surface_xy_tol_A is None
            else float(adsorbate_surface_xy_tol_A)
        )
        self.termination_site_xy_tol = float(termination_site_xy_tol)
        self.z_max_support = z_max_support
        self.vertical_offset = vertical_offset
        self.vertical_adjust_step = float(vertical_adjust_step)
        self.max_vertical_adjust = float(max_vertical_adjust)
        self.surface_layer_tol = float(surface_layer_tol)
        self.bridge_cutoff = bridge_cutoff
        self.bridge_cutoff_scale = 1.15
        self.termination_clearance = float(termination_clearance)
        self.relax = relax
        self.traj_file = traj_file
        self.accepted_traj_file = accepted_traj_file
        self.rejected_traj_file = rejected_traj_file
        self.attempted_traj_file = attempted_traj_file
        self.debug_traj_interval = int(debug_traj_interval)
        if self.debug_traj_interval < 1:
            raise ValueError("debug_traj_interval must be >= 1.")
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.diagnostics_log = bool(diagnostics_log)
        self.diagnostics_top_n = int(diagnostics_top_n)
        if self.diagnostics_top_n < 1:
            raise ValueError("diagnostics_top_n must be >= 1.")
        self.thermo_file = thermo_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self._site_registry: Optional[list[dict[str, object]]] = None
        self._adsorption_channel_registry = None
        self._adsorption_channel_lookup = None
        self._reuse_io = False
        self._persistent_traj_writers: dict[str, Trajectory] = {}
        self._persistent_thermo_handles: dict[str, object] = {}
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
        self.enforce_molecular_integrity = bool(enforce_molecular_integrity)
        self.molecular_bond_stretch_factor = float(molecular_bond_stretch_factor)
        self.molecular_bond_abs_tol = float(molecular_bond_abs_tol)
        self.site_region_center_A = center
        self.site_region_radius_A = (
            None if site_region_radius_A is None else float(site_region_radius_A)
        )
        self.site_region_distance_metric = site_region_distance_metric
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
        if self.molecular_bond_stretch_factor <= 1.0:
            raise ValueError("molecular_bond_stretch_factor must be > 1.")
        if self.molecular_bond_abs_tol < 0.0:
            raise ValueError("molecular_bond_abs_tol must be >= 0.")
        if molecular_upright_min_z_A is not None:
            molecular_upright_min_z_A = float(molecular_upright_min_z_A)
        if molecular_upright_atom_indices is None:
            upright_indices: Tuple[int, ...] = ()
        else:
            upright_indices = tuple(int(idx) for idx in molecular_upright_atom_indices)
            for idx in upright_indices:
                if not (0 <= idx < self.adsorbate_size):
                    raise ValueError(
                        "molecular_upright_atom_indices entries must be valid "
                        "adsorbate template indices."
                    )
                if idx == self.adsorbate_anchor_index:
                    raise ValueError(
                        "molecular_upright_atom_indices must not include the anchor index."
                    )
        if self.md_accept_mode == "hamiltonian":
            if self.md_ensemble != "nve":
                raise ValueError(
                    "md_accept_mode='hamiltonian' requires md_ensemble='nve'."
                )
            if not self.md_init_momenta:
                raise ValueError(
                    "md_accept_mode='hamiltonian' requires md_init_momenta=True."
                )
        self._template_bond_limits = self._build_template_bond_limits()
        self.molecular_upright_atom_indices = upright_indices
        self.molecular_upright_min_z_A = (
            0.0 if molecular_upright_min_z_A is None else molecular_upright_min_z_A
        )
        self._hybrid_move_table = self._build_hybrid_move_table()

        self._update_indices()
        # Site identities and proposal counts must not depend on the current
        # puckering/thermal state. Placement heights are still evaluated from
        # the current coordinates of each site's fixed support atoms.
        hop_modes = {
            "channel_hop",
            "site_hop",
            "hop_reorientation",
            "hop_puckering",
            "hop_puckering_reorientation",
            "hybrid",
        }
        if self.move_mode in hop_modes:
            self._site_registry = self._build_site_registry()
        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.n_samples = 0
        self.accepted_moves = 0
        self.total_moves = 0
        self.md_attempted_moves = 0
        self.md_accepted_moves = 0
        self.move_diagnostics = self._empty_move_diagnostics()
        self.sweep = 0
        self._resumed_from_checkpoint = False

        self.atoms.calc = self.calculator
        self.e_old = self.atoms.get_potential_energy()

        if resume:
            self._load_checkpoint()

    def _normalize_template_library(
        self,
        library: Optional[Sequence[Dict[str, object]]],
    ) -> list[dict[str, object]]:
        if library is None:
            raw_specs: list[dict[str, object]] = [
                {
                    "adsorbate": self.adsorbate_template.copy(),
                    "anchor_index": self.adsorbate_anchor_index,
                    "anchor_mode": self.adsorbate_anchor_mode,
                    "anchor_atom_indices": self.adsorbate_anchor_reference_indices,
                    "weight": 1.0,
                    "name": "primary",
                }
            ]
        else:
            raw_specs = [dict(spec) for spec in library]
            if not raw_specs:
                raise ValueError("adsorbate_template_library must not be empty.")

        specs: list[dict[str, object]] = []
        total_weight = 0.0
        for spec_id, raw in enumerate(raw_specs):
            template_value = raw.get("adsorbate", raw.get("template", raw.get("path")))
            if template_value is None:
                raise ValueError("Each adsorbate template library entry needs adsorbate/template/path.")
            template = _load_adsorbate_template(template_value, self.adsorbate_anchor_symbol)
            if len(template) != self.adsorbate_size:
                raise ValueError("All adsorbate templates must have the same number of atoms.")
            symbols = tuple(template.get_chemical_symbols())
            if symbols != self.adsorbate_symbols:
                raise ValueError(
                    "All adsorbate templates must use the same atom ordering as the primary template."
                )

            anchor_index = int(raw.get("anchor_index", self.adsorbate_anchor_index))
            if not (0 <= anchor_index < len(template)):
                raise ValueError("adsorbate template anchor_index is out of range.")
            anchor_cfg = raw.get("anchor", None)
            anchor_mode = raw.get("anchor_mode", self.adsorbate_anchor_mode)
            anchor_atom_indices = raw.get(
                "anchor_atom_indices",
                raw.get("atom_indices", None),
            )
            if isinstance(anchor_cfg, dict):
                anchor_index = int(
                    anchor_cfg.get(
                        "reference_atom_index",
                        anchor_cfg.get("atom_index", anchor_index),
                    )
                )
                if not (0 <= anchor_index < len(template)):
                    raise ValueError("adsorbate template anchor atom index is out of range.")
                anchor_mode = anchor_cfg.get("mode", anchor_mode)
                anchor_atom_indices = anchor_cfg.get(
                    "atom_indices",
                    anchor_cfg.get("indices", anchor_atom_indices),
                )

            anchor_mode = _normalize_anchor_mode(anchor_mode)
            anchor_atom_indices = _normalize_anchor_reference_indices(
                mode=anchor_mode,
                anchor_index=anchor_index,
                atom_indices=anchor_atom_indices,
                size=len(template),
            )
            weight = float(raw.get("weight", 1.0))
            if weight < 0.0:
                raise ValueError("adsorbate template weights must be non-negative.")
            total_weight += weight
            ref_pos = _anchor_reference_position(
                template,
                mode=anchor_mode,
                anchor_index=anchor_index,
                atom_indices=anchor_atom_indices,
            )
            atom_ref = np.asarray(template.positions[anchor_index], dtype=float)
            specs.append(
                {
                    "id": int(spec_id),
                    "name": str(raw.get("name", f"template{spec_id:03d}")),
                    "template": template.copy(),
                    "anchor_index": anchor_index,
                    "anchor_mode": anchor_mode,
                    "anchor_atom_indices": tuple(anchor_atom_indices),
                    "weight": weight,
                    "relative": np.asarray(template.get_positions(), dtype=float) - ref_pos,
                    "atom_anchor_relative": np.asarray(template.get_positions(), dtype=float)
                    - atom_ref,
                }
            )

        if total_weight <= 0.0:
            raise ValueError("At least one adsorbate template weight must be positive.")
        for spec in specs:
            spec["probability"] = float(spec["weight"]) / total_weight
        return specs

    def _template_match_rmsd(
        self,
        group: np.ndarray,
        spec: dict[str, object],
        atoms: Optional[Atoms] = None,
    ) -> float:
        if atoms is None:
            atoms = self.atoms
        group = np.asarray(group, dtype=int)
        anchor_index = int(spec["anchor_index"])
        if not (0 <= anchor_index < len(group)):
            return float("inf")
        anchor_idx = int(group[anchor_index])
        current_rel = np.asarray(
            atoms.get_distances(
                anchor_idx,
                group,
                mic=True,
                vector=True,
            ),
            dtype=float,
        )
        reference = np.asarray(spec["atom_anchor_relative"], dtype=float)
        current_centered = current_rel - np.mean(current_rel, axis=0)
        reference_centered = reference - np.mean(reference, axis=0)
        try:
            u, _, vt = np.linalg.svd(reference_centered.T @ current_centered)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            aligned = reference_centered @ rotation
            return float(np.sqrt(np.mean(np.sum((aligned - current_centered) ** 2, axis=1))))
        except np.linalg.LinAlgError:
            return float(np.sqrt(np.mean(np.sum((reference_centered - current_centered) ** 2, axis=1))))

    def _template_spec_for_group(
        self,
        group: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> dict[str, object]:
        if len(self.adsorbate_template_library) == 1:
            return self.adsorbate_template_library[0]
        scored = [
            (self._template_match_rmsd(group, spec, atoms=atoms), spec)
            for spec in self.adsorbate_template_library
        ]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _sample_template_spec(self) -> dict[str, object]:
        if len(self.adsorbate_template_library) == 1:
            return self.adsorbate_template_library[0]
        probabilities = np.asarray(
            [float(spec["probability"]) for spec in self.adsorbate_template_library],
            dtype=float,
        )
        idx = int(self.rng.choice(len(self.adsorbate_template_library), p=probabilities))
        return self.adsorbate_template_library[idx]

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

    def _group_anchor_position(
        self,
        group: np.ndarray,
        atoms: Optional[Atoms] = None,
        *,
        spec: Optional[dict[str, object]] = None,
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms
        group = np.asarray(group, dtype=int)
        if spec is None:
            spec = self._template_spec_for_group(group, atoms=atoms)

        anchor_local = int(spec["anchor_index"])
        anchor_idx = int(group[anchor_local])
        mode = str(spec["anchor_mode"])
        ref_indices = tuple(int(idx) for idx in spec["anchor_atom_indices"])
        if mode == "atom":
            return np.asarray(atoms.positions[anchor_idx], dtype=float).copy()

        vectors = np.asarray(
            atoms.get_distances(
                anchor_idx,
                group[np.asarray(ref_indices, dtype=int)],
                mic=True,
                vector=True,
            ),
            dtype=float,
        )
        unwrapped = np.asarray(atoms.positions[anchor_idx], dtype=float) + vectors
        if mode == "centroid":
            return np.mean(unwrapped, axis=0)
        masses = np.asarray(
            [atomic_masses[atomic_numbers[atoms[int(group[idx])].symbol]] for idx in ref_indices],
            dtype=float,
        )
        if not np.all(np.isfinite(masses)) or float(np.sum(masses)) <= 0.0:
            return np.mean(unwrapped, axis=0)
        return np.average(unwrapped, axis=0, weights=masses)

    def _current_group_relative_positions(
        self, group: np.ndarray, atoms: Optional[Atoms] = None
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms
        group = np.asarray(group, dtype=int)
        spec = self._template_spec_for_group(group, atoms=atoms)
        anchor_local = int(spec["anchor_index"])
        anchor_idx = int(group[anchor_local])
        vectors = np.asarray(
            atoms.get_distances(
                anchor_idx,
                group,
                mic=True,
                vector=True,
            ),
            dtype=float,
        )
        unwrapped = np.asarray(atoms.positions[anchor_idx], dtype=float) + vectors
        return unwrapped - self._group_anchor_position(group, atoms=atoms, spec=spec)

    def _template_group_relative_positions(
        self, group: np.ndarray, atoms: Optional[Atoms] = None
    ) -> np.ndarray:
        """Return template-relative adsorbate coordinates for compatible groups."""

        if atoms is None:
            atoms = self.atoms
        group = np.asarray(group, dtype=int)
        if len(group) != self.adsorbate_size:
            return self._current_group_relative_positions(group, atoms=atoms)

        group_symbols = tuple(atoms[int(i)].symbol for i in group)
        if group_symbols != self.adsorbate_symbols:
            return self._current_group_relative_positions(group, atoms=atoms)

        template_positions = np.asarray(
            self.adsorbate_template.get_positions(),
            dtype=float,
        )
        spec = self._template_spec_for_group(group, atoms=atoms)
        return np.asarray(spec["relative"], dtype=float).copy()

    def _sample_template_group_relative_positions(self) -> np.ndarray:
        spec = self._sample_template_spec()
        return np.asarray(spec["relative"], dtype=float).copy()

    def _molecular_move_relative_positions(
        self,
        group: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        if self.is_molecular_adsorbate:
            return self._template_group_relative_positions(group, atoms=atoms)
        return self._current_group_relative_positions(group, atoms=atoms)

    def _group_orientation_is_valid(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if not self.molecular_upright_atom_indices:
            return True
        if atoms is None:
            atoms = self.atoms

        group = np.asarray(group, dtype=int)
        trial_positions = np.asarray(trial_positions, dtype=float)
        if len(group) != self.adsorbate_size or len(trial_positions) != len(group):
            return True
        group_symbols = tuple(atoms[int(i)].symbol for i in group)
        if group_symbols != self.adsorbate_symbols:
            return True

        anchor_local = self._group_anchor_local_index(group, atoms=atoms)
        side_sign = 1.0 if self.surface_side == "top" else -1.0
        spec = self._template_spec_for_group(group, atoms=atoms)
        if str(spec["anchor_mode"]) == "atom":
            anchor_z = float(trial_positions[anchor_local, 2])
        else:
            ref_indices = np.asarray(spec["anchor_atom_indices"], dtype=int)
            if str(spec["anchor_mode"]) == "centroid":
                anchor_z = float(np.mean(trial_positions[ref_indices, 2]))
            else:
                weights = np.asarray(
                    [
                        atomic_masses[atomic_numbers[atoms[int(group[idx])].symbol]]
                        for idx in ref_indices
                    ],
                    dtype=float,
                )
                anchor_z = float(np.average(trial_positions[ref_indices, 2], weights=weights))
        min_gap = float(self.molecular_upright_min_z_A) - 1e-12
        for local_idx in self.molecular_upright_atom_indices:
            gap = side_sign * (float(trial_positions[int(local_idx), 2]) - anchor_z)
            if gap < min_gap:
                return False
        return True

    def _molecular_orientations_are_valid(self, atoms: Atoms) -> bool:
        if not self.molecular_upright_atom_indices:
            return True
        for group in self._adsorbate_groups_for_atoms(atoms):
            group = np.asarray(group, dtype=int)
            if not self._group_orientation_is_valid(
                group,
                atoms.positions[group],
                atoms=atoms,
            ):
                return False
        return True

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

        excluded = set(self.substrate_elements) | set(self.functional_elements)
        unique_adsorbate_symbols = set(self.adsorbate_symbols) - excluded
        if self.allow_ambiguous_empty_adsorbates and unique_adsorbate_symbols:
            if not any(atom.symbol in unique_adsorbate_symbols for atom in atoms):
                return []

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
        self._update_indices()

    def _slab_atoms_for_site_registry(self, atoms: Optional[Atoms] = None) -> Atoms:
        if atoms is None:
            atoms = self.atoms
        groups = self._adsorbate_groups_for_atoms(atoms)
        if not groups:
            return atoms

        slab_atoms = atoms.copy()
        for group in sorted(
            (np.asarray(group, dtype=int) for group in groups),
            key=lambda group: int(np.min(group)),
            reverse=True,
        ):
            for idx in np.sort(group)[::-1]:
                del slab_atoms[int(idx)]
        return slab_atoms

    @classmethod
    def from_clean_surface(
        cls,
        atoms: Atoms,
        calculator: Any,
        adsorbate_element: str = "H",
        adsorbate: Optional[Union[str, Atoms]] = None,
        adsorbate_anchor_index: int = 0,
        adsorbate_anchor_mode: str = "atom",
        adsorbate_anchor_atom_indices: Optional[Sequence[int]] = None,
        substrate_elements: Tuple[str, ...] = ("Ti", "C"),
        top_layer_element: str = "Ti",
        site_elements: Optional[Union[str, Sequence[str]]] = None,
        surface_side: str = "top",
        functional_elements: Optional[Tuple[str, ...]] = None,
        coverage: float = 1.0,
        site_type: Union[str, Sequence[str]] = "fcc",
        min_clearance: float = 0.8,
        adsorbate_surface_clearance_A: float = 0.0,
        adsorbate_surface_xy_tol_A: Optional[float] = None,
        site_match_tol: float = 0.6,
        support_xy_tol: Optional[float] = None,
        termination_site_xy_tol: Optional[float] = None,
        surface_layer_tol: float = 0.5,
        termination_clearance: float = 0.75,
        bridge_cutoff: Optional[float] = None,
        vertical_offset: float = 1.8,
        detach_tol: float = 3.0,
        seed: int = 81,
        initial_traj_file: str = "adsorbate_cmc_initial.traj",
        **kwargs,
    ) -> "AdsorbateCMC":
        if support_xy_tol is None:
            support_xy_tol = max(1.2, 2.5 * float(site_match_tol))
        if termination_site_xy_tol is None:
            termination_site_xy_tol = float(support_xy_tol)
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
            termination_site_xy_tol=termination_site_xy_tol,
            vertical_offset=vertical_offset,
            termination_elements=functional_elements,
            min_termination_dist=termination_clearance,
        )
        atoms_with_ads = _place_adsorbate_template(
            atoms,
            adsorbate_template,
            anchor_index=int(adsorbate_anchor_index),
            anchor_mode=adsorbate_anchor_mode,
            anchor_atom_indices=adsorbate_anchor_atom_indices,
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
            adsorbate_anchor_mode=adsorbate_anchor_mode,
            adsorbate_anchor_atom_indices=adsorbate_anchor_atom_indices,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            top_layer_element=top_layer_element,
            site_elements=site_elements,
            surface_side=surface_side,
            coverage=coverage,
            site_type=site_type,
            min_clearance=min_clearance,
            adsorbate_surface_clearance_A=adsorbate_surface_clearance_A,
            adsorbate_surface_xy_tol_A=adsorbate_surface_xy_tol_A,
            site_match_tol=site_match_tol,
            support_xy_tol=support_xy_tol,
            termination_site_xy_tol=termination_site_xy_tol,
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
            "accepted_moves": self.accepted_moves,
            "total_moves": self.total_moves,
            "md_attempted_moves": self.md_attempted_moves,
            "md_accepted_moves": self.md_accepted_moves,
            "site_registry": self._site_registry,
            "puckering_reference_positions": self._puckering_reference_positions,
            "puckering_coordinate_version": 2,
            "adsorption_channel_version": 1,
        }
        if self.diagnostics_enabled:
            state["move_diagnostics"] = self.move_diagnostics
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
        self.accepted_moves = int(state.get("accepted_moves", self.accepted_moves))
        self.total_moves = int(state.get("total_moves", self.total_moves))
        self.md_attempted_moves = int(
            state.get("md_attempted_moves", self.md_attempted_moves)
        )
        self.md_accepted_moves = int(
            state.get("md_accepted_moves", self.md_accepted_moves)
        )
        reference_positions = state.get("puckering_reference_positions")
        if reference_positions is not None:
            reference_positions = np.asarray(reference_positions, dtype=float)
            if reference_positions.shape == self._puckering_reference_positions.shape:
                self._puckering_reference_positions = reference_positions.copy()
        checkpoint_registry = state.get("site_registry")
        if isinstance(checkpoint_registry, list):
            self._site_registry = checkpoint_registry
        self._adsorption_channel_registry = None
        self._adsorption_channel_lookup = None
        self._puckering_surface_support_indices_cache = None
        self._puckering_local_reference_cache.clear()
        if int(state.get("puckering_coordinate_version", 1)) < 2 and (
            self.puckering_prob > 0.0
            or self.hop_puckering_prob > 0.0
            or self.hop_puckering_reorientation_prob > 0.0
            or self.move_mode
            in {"puckering", "hop_puckering", "hop_puckering_reorientation"}
        ):
            logger.warning(
                "Checkpoint predates the local puckering coordinate. Resume is "
                "supported for inspection, but a fresh production trajectory is "
                "required for a consistent proposal kernel."
            )
        if int(state.get("adsorption_channel_version", 0)) < 1 and (
            self.move_mode == "channel_hop"
            or (self.move_mode == "hybrid" and self.channel_hop_prob > 0.0)
        ):
            logger.warning(
                "Checkpoint predates the adsorption-channel proposal kernel. "
                "Resume is supported for inspection, but a fresh production "
                "trajectory is required for a consistent Markov chain."
            )
        if self.diagnostics_enabled:
            self.move_diagnostics = self._normalize_move_diagnostics(
                state.get("move_diagnostics", self.move_diagnostics)
            )
        self._update_indices()
        self._resumed_from_checkpoint = True
        logger.info(f"[{self.T:.0f}K] Resumed adsorbate MC from checkpoint.")

    def _metropolis_accept(self, delta_e: float, beta: Optional[float] = None) -> bool:
        if delta_e < 0.0:
            return True
        if beta is None:
            beta = 1.0 / (KB_EV_PER_K * self.T)
        return self.rng.random() < np.exp(-delta_e * beta)

    def _metropolis_acceptance_probability(
        self,
        delta_e: float,
        *,
        beta: Optional[float] = None,
    ) -> float:
        if delta_e <= 0.0:
            return 1.0
        if beta is None:
            beta = 1.0 / (KB_EV_PER_K * self.T)
        return float(np.exp(-float(delta_e) * float(beta)))

    def _build_template_bond_limits(self) -> list[tuple[int, int, float]]:
        if not self.is_molecular_adsorbate:
            return []

        positions = np.asarray(self.adsorbate_template.get_positions(), dtype=float)
        symbols = list(self.adsorbate_template.get_chemical_symbols())
        limits: list[tuple[int, int, float]] = []
        for i in range(len(symbols)):
            ri = float(covalent_radii[atomic_numbers[symbols[i]]])
            for j in range(i + 1, len(symbols)):
                rj = float(covalent_radii[atomic_numbers[symbols[j]]])
                template_dist = float(np.linalg.norm(positions[j] - positions[i]))
                if template_dist <= 1.2 * (ri + rj):
                    max_dist = max(
                        template_dist * self.molecular_bond_stretch_factor,
                        template_dist + self.molecular_bond_abs_tol,
                    )
                    limits.append((i, j, max_dist))
        return limits

    def _group_matches_template_connectivity(
        self,
        group: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if atoms is None:
            atoms = self.atoms
        if not self.is_molecular_adsorbate or not self.enforce_molecular_integrity:
            return True
        if not self._template_bond_limits:
            return True

        group = np.asarray(group, dtype=int)
        if len(group) != self.adsorbate_size:
            return False

        for local_i, local_j, max_dist in self._template_bond_limits:
            dist = float(
                atoms.get_distance(int(group[local_i]), int(group[local_j]), mic=True)
            )
            if dist > max_dist:
                return False
        return True

    def _molecular_adsorbates_are_intact(self, atoms: Optional[Atoms] = None) -> bool:
        if atoms is None:
            atoms = self.atoms
        if not self.is_molecular_adsorbate or not self.enforce_molecular_integrity:
            return True

        groups = self._adsorbate_groups_for_atoms(atoms)
        if not groups:
            return True
        return all(
            self._group_matches_template_connectivity(
                np.asarray(group, dtype=int), atoms=atoms
            )
            for group in groups
        )

    def _empty_move_diagnostics(self) -> Dict[str, Dict[str, int]]:
        return {
            "selected_by_move": {},
            "attempted_by_move": {},
            "null_by_reason": {},
            "null_by_move_reason": {},
            "energy_tested_by_move": {},
            "accepted_by_move": {},
            "rejected_by_move": {},
            "rejected_by_reason": {},
            "rejected_by_move_reason": {},
        }

    def _normalize_move_diagnostics(self, value: object) -> Dict[str, Dict[str, int]]:
        diagnostics = self._empty_move_diagnostics()
        if not isinstance(value, dict):
            return diagnostics
        for key in diagnostics:
            raw = value.get(key, {})
            if isinstance(raw, dict):
                diagnostics[key] = {str(k): int(v) for k, v in raw.items()}
        return diagnostics

    def _increment_diagnostic(self, section: str, key: str) -> None:
        if not self.diagnostics_enabled:
            return
        bucket = self.move_diagnostics.setdefault(section, {})
        bucket[str(key)] = int(bucket.get(str(key), 0)) + 1

    def _record_attempted_proposal(self, proposal: MoveProposal) -> None:
        self._increment_diagnostic("attempted_by_move", proposal.move_name)

    def _record_selected_move(self, move_name: str) -> None:
        self._increment_diagnostic("selected_by_move", move_name)

    def _record_null_move(self, move_name: str, reason: str) -> None:
        reason = str(reason)
        self._increment_diagnostic("null_by_reason", reason)
        self._increment_diagnostic(
            "null_by_move_reason",
            f"{move_name}:{reason}",
        )

    def _record_energy_tested_proposal(self, proposal: MoveProposal) -> None:
        self._increment_diagnostic("energy_tested_by_move", proposal.move_name)

    def _record_accepted_proposal(self, proposal: MoveProposal) -> None:
        self._increment_diagnostic("accepted_by_move", proposal.move_name)

    def _record_rejected_proposal(
        self,
        proposal: MoveProposal,
        reason: str,
    ) -> None:
        reason = str(reason)
        self._increment_diagnostic("rejected_by_move", proposal.move_name)
        self._increment_diagnostic("rejected_by_reason", reason)
        self._increment_diagnostic(
            "rejected_by_move_reason",
            f"{proposal.move_name}:{reason}",
        )

    def _top_diagnostics(self, section: str, *, limit: int = 3) -> str:
        if not self.diagnostics_enabled:
            return "disabled"
        bucket = self.move_diagnostics.get(section, {})
        if not bucket:
            return "none"
        items = sorted(bucket.items(), key=lambda item: (-int(item[1]), str(item[0])))
        return ",".join(f"{key}={value}" for key, value in items[:limit])

    def _diagnostics_log_suffix(self) -> str:
        if not (self.diagnostics_enabled and self.diagnostics_log):
            return ""
        top_n = int(self.diagnostics_top_n)
        return (
            f" | Moves: {self._top_diagnostics('selected_by_move', limit=top_n)}"
            f" | Nulls: {self._top_diagnostics('null_by_reason', limit=top_n)}"
            f" | Rejects: {self._top_diagnostics('rejected_by_reason', limit=top_n)}"
        )

    def _diagnostic_stats(self) -> Dict[str, Dict[str, int]]:
        if not self.diagnostics_enabled:
            return self._empty_move_diagnostics()
        return {
            key: dict(value)
            for key, value in self._normalize_move_diagnostics(
                self.move_diagnostics
            ).items()
        }

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
        if self.move_mode in ("channel_hop", "site_hop", "hybrid"):
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
        self,
        new_xy: np.ndarray,
        exclude_indices: Optional[Sequence[int]] = None,
        atoms: Optional[Atoms] = None,
    ) -> Optional[float]:
        if atoms is None:
            atoms = self.atoms

        all_pos = atoms.get_positions()
        surface_indices = self._surface_reference_indices_for_atoms(
            atoms,
            np.asarray([], dtype=int),
        )
        if exclude_indices is not None:
            exclude = np.asarray([int(idx) for idx in exclude_indices], dtype=int)
            surface_indices = surface_indices[~np.isin(surface_indices, exclude)]
        if surface_indices.size == 0:
            return None

        dxy = self._xy_distances(new_xy, all_pos[surface_indices, :2], atoms=atoms)
        support_indices = surface_indices[dxy < self.support_xy_tol]
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
        if float(np.min(dists)) < self.min_clearance:
            return False
        return self._group_stays_on_surface_side(
            group, trial_positions, atoms=atoms
        )

    def _surface_reference_indices_for_atoms(
        self,
        atoms: Atoms,
        group: np.ndarray,
    ) -> np.ndarray:
        tags = np.asarray(atoms.get_tags(), dtype=int)
        ads_mask = np.zeros(len(atoms), dtype=bool)
        if len(tags) == len(atoms):
            ads_mask |= tags >= ADSORBATE_TAG_OFFSET
        else:
            try:
                for ads_group in self._adsorbate_groups_for_atoms(atoms):
                    ads_mask[np.asarray(ads_group, dtype=int)] = True
            except ValueError:
                pass
        ads_mask[np.asarray(group, dtype=int)] = True
        return np.where(~ads_mask)[0]

    def _group_stays_on_surface_side(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if atoms is None:
            atoms = self.atoms

        group = np.asarray(group, dtype=int)
        trial_positions = np.asarray(trial_positions, dtype=float)
        reference_indices = self._surface_reference_indices_for_atoms(atoms, group)
        if reference_indices.size == 0 or trial_positions.size == 0:
            return True

        reference_positions = atoms.get_positions()[reference_indices]
        dxy = np.vstack(
            [
                self._xy_distances(
                    trial_position[:2],
                    reference_positions[:, :2],
                    atoms=atoms,
                )
                for trial_position in trial_positions
            ]
        )
        local_mask = dxy < self.adsorbate_surface_xy_tol_A
        if not np.any(local_mask):
            return True

        side_sign = 1.0 if self.surface_side == "top" else -1.0
        side_gap = side_sign * (
            trial_positions[:, None, 2] - reference_positions[None, :, 2]
        )
        min_gap = self.adsorbate_surface_clearance_A - 1e-12
        return not bool(np.any(local_mask & (side_gap < min_gap)))

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
            dz = deltas[0, :, 2]
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
        dz = -deltas[:, :, 2]

        side_sign = 1.0 if self.surface_side == "top" else -1.0
        support_mask = (
            (dxy < support_xy_tol)
            & ((side_sign * dz) > 0)
            & ((side_sign * dz) < z_max_support)
        )
        return bool(np.any(np.sum(support_mask, axis=1) == 0))

    def _build_site_registry(self) -> list[dict[str, object]]:
        slab_atoms = self._slab_atoms_for_site_registry()
        registry = build_surface_site_registry(
            slab_atoms,
            site_elements=self.site_elements,
            substrate_elements=self.substrate_elements,
            surface_side=self.surface_side,
            site_types=self.site_types,
            layer_tol=self.surface_layer_tol,
            xy_tol=self.site_match_tol,
            bridge_cutoff=self.bridge_cutoff,
            bridge_cutoff_scale=self.bridge_cutoff_scale,
            support_xy_tol=self.support_xy_tol,
            termination_site_xy_tol=self.termination_site_xy_tol,
            vertical_offset=self.vertical_offset,
            termination_elements=self.functional_elements,
            min_termination_dist=self.termination_clearance,
        )
        filtered = self._filter_site_registry_to_region(registry, slab_atoms)
        for site_id, site in enumerate(filtered):
            site["site_id"] = int(site_id)
        return filtered

    def _get_site_registry(self) -> list[dict[str, object]]:
        if self._site_registry is None:
            self._site_registry = self._build_site_registry()
        return self._site_registry

    def _filter_site_registry_to_region(
        self,
        registry: Sequence[dict[str, object]],
        atoms: Atoms,
    ) -> list[dict[str, object]]:
        if self.site_region_center_A is None or self.site_region_radius_A is None:
            return [dict(site) for site in registry]
        return [
            dict(site)
            for site in registry
            if self._point_within_site_region(
                np.array(
                    [
                        float(np.asarray(site["xy"], dtype=float)[0]),
                        float(np.asarray(site["xy"], dtype=float)[1]),
                        float(site.get("suggested_z_A", self.site_region_center_A[2])),
                    ],
                    dtype=float,
                ),
                atoms=atoms,
            )
        ]

    def _point_within_site_region(
        self,
        point: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> bool:
        if self.site_region_center_A is None or self.site_region_radius_A is None:
            return True
        if atoms is None:
            atoms = self.atoms
        point = np.asarray(point, dtype=float)
        center = np.asarray(self.site_region_center_A, dtype=float)
        if self.site_region_distance_metric == "xy":
            point = point.copy()
            center = center.copy()
            point[2] = 0.0
            center[2] = 0.0
        delta = point - center
        if np.any(atoms.get_pbc()):
            try:
                delta = get_distances(
                    center.reshape(1, 3),
                    point.reshape(1, 3),
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )[0][0, 0]
                if self.site_region_distance_metric == "xy":
                    delta[2] = 0.0
            except Exception:
                pass
        return float(np.linalg.norm(delta)) <= float(self.site_region_radius_A)

    def _anchors_within_site_region(self, atoms: Optional[Atoms] = None) -> bool:
        if self.site_region_center_A is None or self.site_region_radius_A is None:
            return True
        if atoms is None:
            atoms = self.atoms
        for group in self._adsorbate_groups_for_atoms(atoms):
            anchor_pos = self._group_anchor_position(
                np.asarray(group, dtype=int),
                atoms=atoms,
            )
            if not self._point_within_site_region(anchor_pos, atoms=atoms):
                return False
        return True

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

    def _adjust_trial_positions_vertically(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> Optional[np.ndarray]:
        if atoms is None:
            atoms = self.atoms

        adjusted = np.asarray(trial_positions, dtype=float).copy()
        if adjusted.ndim != 2 or adjusted.shape[1] != 3:
            raise ValueError("trial_positions must have shape (n_atoms, 3).")

        direction = 1.0 if self.surface_side == "top" else -1.0
        total_adjust = 0.0

        while True:
            if self._group_positions_are_valid(group, adjusted, atoms=atoms) and (
                self._group_clears_terminations(group, adjusted, atoms=atoms)
            ):
                return adjusted

            if total_adjust >= self.max_vertical_adjust:
                return None

            dz = min(
                float(self.vertical_adjust_step),
                float(self.max_vertical_adjust) - total_adjust,
            )
            adjusted[:, 2] += direction * dz
            total_adjust += dz

    def _validate_trial_atoms(self, atoms_trial: Atoms) -> Optional[str]:
        if self.has_detached_functional_groups(
            atoms_trial,
            detach_tol=self.detach_tol,
        ):
            return "detached_functional_group"
        if self.has_afloat_adsorbates(
            atoms_trial,
            support_xy_tol=self.support_xy_tol,
            z_max_support=self.z_max_support,
        ):
            return "afloat_adsorbate"
        if not self._molecular_adsorbates_are_intact(atoms_trial):
            return "molecular_integrity"
        if not self._molecular_orientations_are_valid(atoms_trial):
            return "molecular_orientation"
        if not self._anchors_within_site_region(atoms_trial):
            return "site_region"
        return None

    def _write_debug_atoms(
        self,
        writer: Optional[Trajectory],
        atoms: Atoms,
        write_debug_frame: bool,
        *,
        proposal: MoveProposal,
        event: str,
        reason: Optional[str] = None,
    ) -> None:
        if writer is None or not write_debug_frame:
            return
        frame = atoms.copy()
        frame.info["mc_event"] = event
        frame.info["mc_move_name"] = proposal.move_name
        frame.info["mc_is_md"] = bool(proposal.is_md)
        if reason is not None:
            frame.info["mc_reject_reason"] = reason
        for key, value in proposal.metadata.items():
            frame.info[f"mc_{key}"] = value
        writer.write(frame)

    def _write_sampled_state(self, writer: Trajectory) -> None:
        frame = self.atoms.copy()
        frame.info["mc_event"] = "sampled"
        frame.info["mc_sweep"] = int(self.sweep)
        frame.info["mc_energy_eV"] = float(self.e_old)
        writer.write(frame)

    def _with_energy_metadata(
        self,
        proposal: MoveProposal,
        *,
        energy_eV: float,
        delta_e_eV: float,
        acceptance_delta_eV: float,
        beta: float,
        accepted: bool,
        current_energy_eV: Optional[float] = None,
        delta_h_eV: Optional[float] = None,
        acceptance_mode: str = "potential",
    ) -> MoveProposal:
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "energy_eV": float(energy_eV),
                "delta_e_eV": float(delta_e_eV),
                "acceptance_delta_eV": float(acceptance_delta_eV),
                "accept_prob": self._metropolis_acceptance_probability(
                    acceptance_delta_eV,
                    beta=beta,
                ),
                "accepted": bool(accepted),
                "acceptance_mode": str(acceptance_mode),
            }
        )
        if current_energy_eV is not None:
            metadata["current_energy_eV"] = float(current_energy_eV)
        if delta_h_eV is not None:
            metadata["delta_h_eV"] = float(delta_h_eV)
        return MoveProposal(
            atoms=proposal.atoms,
            move_name=proposal.move_name,
            is_md=proposal.is_md,
            delta_e=proposal.delta_e,
            delta_h=proposal.delta_h,
            metadata=metadata,
        )

    def _relax_proposal(
        self,
        proposal: MoveProposal,
        *,
        move_ind: Sequence[int],
    ) -> tuple[Optional[MoveProposal], Optional[str]]:
        if proposal.is_md or not self.relax:
            return proposal, None

        atoms_trial, converged = self.relax_structure(
            proposal.atoms,
            move_ind=move_ind,
        )
        relaxed = MoveProposal(
            atoms=atoms_trial,
            move_name=proposal.move_name,
            is_md=proposal.is_md,
            delta_e=proposal.delta_e,
            delta_h=proposal.delta_h,
            metadata={**proposal.metadata, "relaxed": bool(converged)},
        )
        return relaxed, None

    def _process_move_proposal(
        self,
        proposal: MoveProposal,
        *,
        beta: float,
        move_ind: Sequence[int],
        write_debug_frame: bool,
        attempted_writer: Optional[Trajectory],
        accepted_writer: Optional[Trajectory],
        rejected_writer: Optional[Trajectory],
    ) -> None:
        self._record_attempted_proposal(proposal)
        if proposal.is_md:
            self._record_energy_tested_proposal(proposal)
        self._write_debug_atoms(
            attempted_writer,
            proposal.atoms,
            write_debug_frame,
            proposal=proposal,
            event="attempted",
        )

        proposal, reject_reason = self._relax_proposal(proposal, move_ind=move_ind)
        if proposal is None:
            return
        if reject_reason is None:
            reject_reason = self._validate_trial_atoms(proposal.atoms)
        if reject_reason is not None:
            self._record_rejected_proposal(proposal, reject_reason)
            self._write_debug_atoms(
                rejected_writer,
                proposal.atoms,
                write_debug_frame,
                proposal=proposal,
                event="rejected",
                reason=reject_reason,
            )
            return

        if proposal.is_md:
            delta_e = float(proposal.delta_e if proposal.delta_e is not None else 0.0)
            delta_h = float(proposal.delta_h if proposal.delta_h is not None else delta_e)
            md_delta = delta_h if self.md_accept_mode == "hamiltonian" else delta_e
            e_current = float(self.e_old)
            e_new = e_current + delta_e
            accepted = self._metropolis_accept(md_delta, beta=beta)
            annotated = self._with_energy_metadata(
                proposal,
                energy_eV=e_new,
                delta_e_eV=delta_e,
                delta_h_eV=delta_h,
                acceptance_delta_eV=md_delta,
                beta=beta,
                accepted=accepted,
                current_energy_eV=e_current,
                acceptance_mode=self.md_accept_mode,
            )
            if accepted:
                self.e_old += delta_e
                self.accepted_moves += 1
                self.md_accepted_moves += 1
                self._record_accepted_proposal(annotated)
                self.atoms.positions = proposal.atoms.positions
                self.atoms.cell = proposal.atoms.cell
                self._write_debug_atoms(
                    accepted_writer,
                    self.atoms,
                    write_debug_frame,
                    proposal=annotated,
                    event="accepted",
                )
            else:
                self._record_rejected_proposal(annotated, "metropolis")
                self._write_debug_atoms(
                    rejected_writer,
                    proposal.atoms,
                    write_debug_frame,
                    proposal=annotated,
                    event="rejected",
                    reason="metropolis",
                )
            return

        self._record_energy_tested_proposal(proposal)
        e_new = self.get_potential_energy(proposal.atoms)
        delta_e = e_new - self.e_old
        e_current = float(self.e_old)
        accepted = self._metropolis_accept(delta_e, beta=beta)
        annotated = self._with_energy_metadata(
            proposal,
            energy_eV=e_new,
            delta_e_eV=delta_e,
            acceptance_delta_eV=delta_e,
            beta=beta,
            accepted=accepted,
            current_energy_eV=e_current,
            acceptance_mode="potential",
        )
        if accepted:
            self.atoms = proposal.atoms
            self.atoms.calc = self.calculator
            self.e_old = e_new
            self.accepted_moves += 1
            self._record_accepted_proposal(annotated)
            self._write_debug_atoms(
                accepted_writer,
                self.atoms,
                write_debug_frame,
                proposal=annotated,
                event="accepted",
            )
        else:
            self._record_rejected_proposal(annotated, "metropolis")
            self._write_debug_atoms(
                rejected_writer,
                proposal.atoms,
                write_debug_frame,
                proposal=annotated,
                event="rejected",
                reason="metropolis",
            )

    def _open_optional_traj(self, filename: Optional[str]) -> Optional[Trajectory]:
        if not filename:
            return None
        mode = "a" if os.path.exists(filename) and os.path.getsize(filename) > 0 else "w"
        return Trajectory(filename, mode)

    def _get_traj_writer(self, filename: Optional[str]) -> Optional[Trajectory]:
        if not filename:
            return None
        if not self._reuse_io:
            return self._open_optional_traj(filename)
        writer = self._persistent_traj_writers.get(filename)
        if writer is None:
            writer = self._open_optional_traj(filename)
            if writer is not None:
                self._persistent_traj_writers[filename] = writer
        return writer

    def _get_thermo_handle(self):
        if not self._reuse_io:
            return None
        handle = self._persistent_thermo_handles.get(self.thermo_file)
        if handle is None or handle.closed:
            handle = open(self.thermo_file, "a", buffering=1)
            self._persistent_thermo_handles[self.thermo_file] = handle
        return handle

    def _should_write_debug_traj(self) -> bool:
        return int(self.total_moves) % int(self.debug_traj_interval) == 0

    def close_persistent_io(self) -> None:
        for writer in self._persistent_traj_writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._persistent_traj_writers.clear()
        for handle in self._persistent_thermo_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._persistent_thermo_handles.clear()

    def run(
        self,
        nsweeps: int,
        traj_file: str,
        interval: int = 10,
        sample_interval: int = 1,
        equilibration: int = 0,
        sweeps_are_total: bool = True,
    ) -> Dict[str, float]:
        self.traj_file = traj_file

        target_sweeps = int(nsweeps)
        if not self._resumed_from_checkpoint:
            self.sum_E = 0.0
            self.sum_E_sq = 0.0
            self.n_samples = 0
            self.accepted_moves = 0
            self.total_moves = 0
            self.md_attempted_moves = 0
            self.md_accepted_moves = 0
            self.move_diagnostics = self._empty_move_diagnostics()
        self._resumed_from_checkpoint = False
        remaining_sweeps = (
            max(0, target_sweeps - int(self.sweep))
            if bool(sweeps_are_total)
            else target_sweeps
        )

        if len(self.ads_groups) == 0:
            logger.warning("AdsorbateCMC run started with no adsorbates present.")
        if not self._molecular_adsorbates_are_intact(self.atoms):
            raise RuntimeError(
                "Current molecular adsorbate state violates the template bond graph. "
                "Start from an intact adsorbate or disable molecular-integrity enforcement."
            )

        chunk_mode = not bool(sweeps_are_total)

        traj_writer = self._get_traj_writer(self.traj_file)
        accepted_writer = self._get_traj_writer(self.accepted_traj_file)
        rejected_writer = self._get_traj_writer(self.rejected_traj_file)
        attempted_writer = self._get_traj_writer(self.attempted_traj_file)
        thermo_handle = self._get_thermo_handle()

        for local_sweep in range(remaining_sweeps):
            beta = 1.0 / (KB_EV_PER_K * self.T)
            moves_this_sweep = self._moves_per_sweep()

            for i in range(moves_this_sweep):
                self.total_moves += 1
                write_debug_frame = self._should_write_debug_traj()
                do_md = self.enable_hybrid_md and self.rng.random() < self.md_move_prob
                if do_md:
                    move_name = "md_boost"
                    self._record_selected_move(move_name)
                    self.md_attempted_moves += 1
                    atoms_trial, delta_e, delta_h = self._propose_md_move()
                    if atoms_trial is None:
                        self._record_null_move(move_name, "md_proposal_failed")
                        continue
                    proposal = MoveProposal(
                        atoms=atoms_trial,
                        move_name=move_name,
                        is_md=True,
                        delta_e=delta_e,
                        delta_h=delta_h,
                    )
                else:
                    atoms_trial = self._propose_move()
                    move_name = (
                        getattr(self, "_last_proposal_move_name", None)
                        or self.move_mode
                    )
                    self._record_selected_move(move_name)
                    if atoms_trial is None:
                        self._record_null_move(
                            move_name,
                            getattr(self, "_last_proposal_reject_reason", None)
                            or "proposal_unavailable",
                        )
                        continue
                    proposal = MoveProposal(
                        atoms=atoms_trial,
                        move_name=move_name,
                        metadata=dict(
                            getattr(self, "_last_proposal_metadata", {})
                        ),
                    )

                self._process_move_proposal(
                    proposal,
                    beta=beta,
                    move_ind=(self.sweep, i),
                    write_debug_frame=write_debug_frame,
                    attempted_writer=attempted_writer,
                    accepted_writer=accepted_writer,
                    rejected_writer=rejected_writer,
                )

            self.sweep += 1
            completed_sweep = int(self.sweep)
            local_completed_sweep = local_sweep + 1

            sample_counter = local_completed_sweep if chunk_mode else completed_sweep
            if sample_counter > equilibration and sample_counter % sample_interval == 0:
                self.sum_E += self.e_old
                self.sum_E_sq += self.e_old**2
                self.n_samples += 1

            report_counter = local_completed_sweep if chunk_mode else completed_sweep
            if report_counter % interval == 0:
                self._write_sampled_state(traj_writer)
                if thermo_handle is None:
                    with open(self.thermo_file, "a") as handle:
                        handle.write(f"{self.sweep} {self.e_old:.6f}\n")
                else:
                    thermo_handle.write(f"{self.sweep} {self.e_old:.6f}\n")
                    thermo_handle.flush()

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
                    f"T={self.T:4.0f}K | {completed_sweep:6d} | "
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
                    + self._diagnostics_log_suffix()
                )

            if (
                self.checkpoint_interval > 0
                and self.sweep % self.checkpoint_interval == 0
            ):
                self._save_checkpoint()

        self._save_checkpoint()
        if not self._reuse_io:
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
            "move_diagnostics": self._diagnostic_stats(),
        }
