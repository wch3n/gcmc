"""
utils.py

Utility functions for MC simulation initial configuration generation on periodic slabs,
including robust, PBC-aware hollow-site registry (via padding) and multilayer adsorbate
placement (works for any coverage, e.g., 2.5 ML).

All functions compatible with ASE Atoms objects.
"""

import os
import numpy as np
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
from ase import Atom, Atoms
from ase.geometry import find_mic, get_distances
from ase.io import read
from ase.symbols import string2symbols
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay


def _cluster_axis_values(values: np.ndarray, tol: float) -> List[np.ndarray]:
    """Cluster sorted 1D coordinates using a simple absolute tolerance."""
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []

    order = np.argsort(values)
    sorted_values = values[order]
    clusters: List[List[int]] = [[int(order[0])]]
    cluster_mean = float(sorted_values[0])

    for sorted_idx, original_idx in enumerate(order[1:], start=1):
        value = float(sorted_values[sorted_idx])
        if abs(value - cluster_mean) <= tol:
            clusters[-1].append(int(original_idx))
            cluster_mean = float(np.mean(values[clusters[-1]]))
        else:
            clusters.append([int(original_idx)])
            cluster_mean = value

    return [np.asarray(cluster, dtype=int) for cluster in clusters]


def _select_site_layers_for_coverage(
    n_sites: int,
    coverage: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Return the site indices occupied in each full/partial coverage layer."""
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0.")
    if coverage < 0:
        raise ValueError("coverage must be >= 0.")

    all_site_indices = np.arange(n_sites, dtype=int)
    n_full_layers = int(np.floor(coverage))
    frac_layer = float(coverage - n_full_layers)
    selected_layers = [all_site_indices.copy() for _ in range(n_full_layers)]

    if frac_layer > 1e-8:
        n_partial = int(np.floor(frac_layer * n_sites + 0.5))
        n_partial = min(n_sites, max(0, n_partial))
        if n_partial > 0:
            selected_layers.append(
                np.asarray(rng.choice(n_sites, n_partial, replace=False), dtype=int)
            )

    return selected_layers


def _normalize_site_types(
    site_types: Union[str, Sequence[str]],
) -> Tuple[str, ...]:
    raw = _normalize_symbol_list(site_types)
    normalized = tuple(
        "atop" if str(site).lower() == "top" else str(site).lower() for site in raw
    )
    if "all" in normalized:
        return ("atop", "bridge", "fcc", "hcp")
    allowed = {"atop", "bridge", "fcc", "hcp"}
    if not set(normalized).issubset(allowed):
        raise ValueError(
            "site_types must be drawn from 'atop', 'bridge', 'fcc', 'hcp', or 'all'."
        )
    return normalized


def generate_nonuniform_temperature_grid(
    T_start: float,
    T_end: float,
    n_replicas: int,
    focus_temps: Optional[Sequence[float]] = None,
    focus_weights: Optional[Sequence[float]] = None,
    focus_strength: float = 4.0,
    focus_width: Optional[float] = None,
    grid_space: Literal["temperature", "beta"] = "temperature",
) -> List[float]:
    """
    Generate a monotonic nonuniform temperature grid with optional dense regions.

    The grid always spans [T_start, T_end] and contains exactly `n_replicas` points.
    Density is increased around one or more focus temperatures using Gaussian bumps.
    Baseline spacing can be uniform in temperature or inverse temperature beta=1/T.

    Args:
        T_start: First temperature in the grid.
        T_end: Last temperature in the grid.
        n_replicas: Number of replicas (grid points).
        focus_temps: One or more target temperatures where spacing should be finer.
        focus_weights: Relative strength per focus temperature (same length as focus_temps).
        focus_strength: Global amplification for all focus temperatures.
        focus_width: Width (in K) of local refinement around each focus temperature.
        grid_space: Baseline coordinate for spacing, either "temperature" or "beta".

    Returns:
        List of temperatures ordered from T_start to T_end.
    """
    n_replicas = int(n_replicas)
    if n_replicas < 2:
        raise ValueError("n_replicas must be >= 2.")

    t0 = float(T_start)
    t1 = float(T_end)
    if np.isclose(t0, t1):
        return [t0 for _ in range(n_replicas)]

    lo, hi = (t0, t1) if t0 < t1 else (t1, t0)
    span = hi - lo

    if focus_temps is None:
        focus_list: List[float] = []
    elif np.isscalar(focus_temps):
        focus_list = [float(focus_temps)]
    else:
        focus_list = [float(t) for t in focus_temps]

    if focus_weights is None:
        weight_list = [1.0] * len(focus_list)
    elif np.isscalar(focus_weights):
        weight_list = [float(focus_weights)]
    else:
        weight_list = [float(w) for w in focus_weights]

    if len(weight_list) != len(focus_list):
        raise ValueError("focus_weights must have the same length as focus_temps.")
    if any(w < 0 for w in weight_list):
        raise ValueError("focus_weights must be non-negative.")
    if focus_strength < 0:
        raise ValueError("focus_strength must be non-negative.")
    if grid_space not in {"temperature", "beta"}:
        raise ValueError("grid_space must be either 'temperature' or 'beta'.")
    if grid_space == "beta" and lo <= 0:
        raise ValueError("T_start and T_end must be > 0 when grid_space='beta'.")

    sigma = float(focus_width) if focus_width is not None else 0.08 * span
    if sigma <= 0:
        raise ValueError("focus_width must be > 0.")

    dense_n = max(4000, n_replicas * 250)
    if grid_space == "temperature":
        dense_t = np.linspace(lo, hi, dense_n)
    else:
        # Uniform indexing in beta yields geometric-like spacing in temperature.
        beta_lo = 1.0 / lo
        beta_hi = 1.0 / hi
        dense_beta = np.linspace(beta_lo, beta_hi, dense_n)
        dense_t = 1.0 / dense_beta
    density = np.ones_like(dense_t)

    for t_focus, weight in zip(focus_list, weight_list):
        if weight == 0.0:
            continue
        if t_focus < lo or t_focus > hi:
            continue
        z = (dense_t - t_focus) / sigma
        density += focus_strength * weight * np.exp(-0.5 * z * z)

    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    quantiles = np.linspace(0.0, 1.0, n_replicas)
    temps = np.interp(quantiles, cdf, dense_t)

    # Preserve exact endpoints for robust comparisons and logging.
    temps[0] = lo
    temps[-1] = hi

    if t0 > t1:
        temps = temps[::-1]

    return [float(t) for t in temps]


def get_toplayer_xy(
    atoms: Atoms, element: str = "Ti", z_tol: float = 0.3
) -> np.ndarray:
    """
    Get xy coordinates of top-layer substrate atoms.

    Args:
        atoms: ASE Atoms object.
        element: Symbol of substrate atom (e.g., "Ti").
        z_tol: z range for top layer.

    Returns:
        (N, 2) array of xy coordinates.
    """
    atoms_elem = [atom for atom in atoms if atom.symbol == element]
    if not atoms_elem:
        raise ValueError(f"No atoms with symbol '{element}' in atoms object.")
    z_max = max(atom.position[2] for atom in atoms_elem)
    xy = np.array(
        [atom.position[:2] for atom in atoms_elem if atom.position[2] > z_max - z_tol]
    )
    return xy


def get_hollow_xy(toplayer_xy: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Find hollow sites for a periodic slab by padding the cell with image points.

    Args:
        toplayer_xy: (N,2) array of top-layer xy positions.
        cell: (3,3) simulation cell.

    Returns:
        (M,2) array of unique hollow xy positions within the center cell.
    """
    a1, a2 = cell[0, :2], cell[1, :2]
    # Tile ±1 in x and y (total 9 images).
    image_points = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            shift = i * a1 + j * a2
            image_points.append(toplayer_xy + shift)
    image_points = np.vstack(image_points)

    tri = Delaunay(image_points)
    hollow_xy = []
    for simplex in tri.simplices:
        pts = image_points[simplex]
        center = pts.mean(axis=0)
        # Check whether the center is inside the original cell (fractional coordinates).
        cell_2d = np.array([a1, a2]).T
        try:
            frac = np.linalg.solve(cell_2d, center)
        except np.linalg.LinAlgError:
            continue
        if np.all((frac >= -1e-8) & (frac < 1 - 1e-8)):
            hollow_xy.append(center)
    # Remove duplicates.
    hollow_xy_unique = []
    tol = 1e-5
    for xy in hollow_xy:
        if not any(np.linalg.norm(xy - h) < tol for h in hollow_xy_unique):
            hollow_xy_unique.append(xy)
    return np.array(hollow_xy_unique)


def _normalize_symbol_list(
    elements: Union[str, Sequence[str]],
) -> Tuple[str, ...]:
    if isinstance(elements, str):
        tokens = tuple(token for token in elements.replace(",", " ").split() if token)
        if tokens:
            return tokens
        return tuple(string2symbols(elements))
    return tuple(str(el) for el in elements)


def _cell_2d(cell: np.ndarray) -> np.ndarray:
    return np.column_stack((cell[0, :2], cell[1, :2]))


def _wrap_xy(xy: np.ndarray, cell: np.ndarray) -> np.ndarray:
    cell_2d = _cell_2d(np.asarray(cell, dtype=float))
    frac = np.linalg.solve(cell_2d, np.asarray(xy, dtype=float))
    frac -= np.floor(frac)
    return np.dot(cell_2d, frac)


def _xy_distances(
    xy: np.ndarray,
    points_xy: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
) -> np.ndarray:
    if len(points_xy) == 0:
        return np.array([], dtype=float)
    disp = np.zeros((len(points_xy), 3), dtype=float)
    disp[:, :2] = np.asarray(points_xy, dtype=float) - np.asarray(xy, dtype=float)[None, :]
    disp_mic, _ = find_mic(disp, np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool))
    return np.linalg.norm(disp_mic[:, :2], axis=1)


def _ordered_layer_indices_by_cluster(
    atoms: Atoms,
    elements: Union[str, Sequence[str]],
    *,
    surface_side: Literal["top", "bottom"] = "top",
    layer_tol: float = 0.5,
) -> List[np.ndarray]:
    """
    Return global atom indices grouped into z layers ordered from the chosen surface.

    For MXenes and other rumpled slabs, callers can pass a larger ``layer_tol`` than
    the site-matching tolerance so physically equivalent planes are not spuriously
    split into several pseudo-layers.
    """
    side = str(surface_side).lower()
    if side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'.")
    symbols = _normalize_symbol_list(elements)
    symbol_set = set(symbols)
    indices = np.array(
        [atom.index for atom in atoms if atom.symbol in symbol_set],
        dtype=int,
    )
    if indices.size == 0:
        raise ValueError(f"No atoms with symbols {symbols} found.")

    zvals = atoms.positions[indices, 2]
    clusters = _cluster_axis_values(zvals, tol=layer_tol)
    if not clusters:
        raise RuntimeError("Could not identify any surface layers.")
    clusters = sorted(
        clusters,
        key=lambda cluster: float(np.mean(zvals[np.asarray(cluster, dtype=int)])),
        reverse=(side == "top"),
    )
    return [np.sort(indices[np.asarray(cluster, dtype=int)].astype(int)) for cluster in clusters]


def _midpoint_xy(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
) -> np.ndarray:
    disp = np.zeros((1, 3), dtype=float)
    disp[0] = np.asarray(pos_j, dtype=float) - np.asarray(pos_i, dtype=float)
    disp_mic, _ = find_mic(disp, np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool))
    xy = np.asarray(pos_i, dtype=float)[:2] + 0.5 * disp_mic[0, :2]
    return _wrap_xy(xy, cell)


def _select_surface_atom_indices_by_cluster(
    atoms: Atoms,
    elements: Union[str, Sequence[str]],
    *,
    surface_side: Literal["top", "bottom"] = "top",
    layer_tol: float = 0.5,
    reference_elements: Optional[Union[str, Sequence[str]]] = None,
    reference_layer_tol: Optional[float] = None,
) -> np.ndarray:
    """
    Select exposed surface atoms for site generation.

    If ``reference_elements`` are provided and the requested ``elements`` are part of
    the outer substrate layer, select that single physical layer. If the requested
    ``elements`` are not part of the outer substrate layer (for example terminating O
    atoms on an MXene), fall back to selecting all matching atoms on the requested
    slab side rather than the outermost z-cluster of that species.
    """
    site_symbols = set(_normalize_symbol_list(elements))
    if reference_elements is not None:
        ref_layers = _ordered_layer_indices_by_cluster(
            atoms,
            reference_elements,
            surface_side=surface_side,
            layer_tol=(
                float(reference_layer_tol)
                if reference_layer_tol is not None
                else float(layer_tol)
            ),
        )
        outer_layer = np.asarray(ref_layers[0], dtype=int)
        selected = np.asarray(
            [idx for idx in outer_layer if atoms[int(idx)].symbol in site_symbols],
            dtype=int,
        )
        if selected.size > 0:
            return np.sort(selected.astype(int))

        site_indices = np.asarray(
            [atom.index for atom in atoms if atom.symbol in site_symbols], dtype=int
        )
        if site_indices.size > 0:
            slab_midpoint = float(np.mean(atoms.positions[:, 2]))
            if surface_side == "top":
                side_mask = atoms.positions[site_indices, 2] >= slab_midpoint
            else:
                side_mask = atoms.positions[site_indices, 2] <= slab_midpoint
            side_indices = site_indices[side_mask]
            if side_indices.size > 0:
                side_z = atoms.positions[side_indices, 2]
                order = np.argsort(side_z)
                if surface_side == "top":
                    order = order[::-1]
                ordered_indices = side_indices[order]
                ordered_z = side_z[order]
                if len(ordered_z) <= 1:
                    return np.sort(ordered_indices.astype(int))

                gaps = np.abs(np.diff(ordered_z))
                split_tol = 0.5
                large_gap_positions = np.where(gaps > split_tol)[0]
                if large_gap_positions.size > 0:
                    split_at = int(large_gap_positions[0]) + 1
                    selected = ordered_indices[:split_at]
                    return np.sort(selected.astype(int))
                return np.sort(ordered_indices.astype(int))

    return np.sort(
        np.asarray(
            _ordered_layer_indices_by_cluster(
                atoms,
                elements,
                surface_side=surface_side,
                layer_tol=layer_tol,
            )[0],
            dtype=int,
        )
    )


def _select_surface_atom_indices_via_terminations(
    atoms: Atoms,
    elements: Union[str, Sequence[str]],
    termination_elements: Union[str, Sequence[str]],
    *,
    substrate_elements: Optional[Union[str, Sequence[str]]] = None,
    surface_side: Literal["top", "bottom"] = "top",
    layer_tol: float = 0.5,
    lateral_tol: float = 2.5,
) -> np.ndarray:
    """
    Identify exposed substrate/site atoms by matching exposed terminations above them.

    This is useful for terminated MXenes where rough thermal disorder makes pure z-layer
    clustering of the exposed metal sublattice unreliable.
    """
    side = str(surface_side).lower()
    if side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'.")

    site_symbols = set(_normalize_symbol_list(elements))
    term_symbols = tuple(_normalize_symbol_list(termination_elements))
    if not site_symbols or not term_symbols:
        return np.empty((0,), dtype=int)

    termination_indices = _select_surface_atom_indices_by_cluster(
        atoms,
        term_symbols,
        surface_side=side,
        layer_tol=layer_tol,
        reference_elements=substrate_elements,
        reference_layer_tol=(
            max(float(layer_tol), 0.9) if substrate_elements is not None else None
        ),
    )
    if termination_indices.size == 0:
        return np.empty((0,), dtype=int)

    slab_midpoint = float(np.mean(atoms.positions[:, 2]))
    candidate_indices = np.asarray(
        [
            atom.index
            for atom in atoms
            if atom.symbol in site_symbols
            and (
                atom.position[2] >= slab_midpoint
                if side == "top"
                else atom.position[2] <= slab_midpoint
            )
        ],
        dtype=int,
    )
    if candidate_indices.size == 0:
        return np.empty((0,), dtype=int)

    term_pos = atoms.positions[termination_indices]
    cand_pos = atoms.positions[candidate_indices]
    deltas = get_distances(term_pos, cand_pos, cell=atoms.cell, pbc=atoms.pbc)[0]
    dxy = np.linalg.norm(deltas[:, :, :2], axis=2)
    dz = term_pos[:, None, 2] - cand_pos[None, :, 2]
    side_sign = 1.0 if side == "top" else -1.0
    valid = (side_sign * dz) > 0.0

    large_cost = 1.0e6
    cost = np.where(valid, dxy, large_cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    keep = cost[row_ind, col_ind] < min(float(lateral_tol), large_cost)
    if not np.any(keep):
        return np.empty((0,), dtype=int)

    return np.sort(np.unique(candidate_indices[col_ind[keep]]).astype(int))


def classify_hollow_sites_on_surface(
    atoms: Atoms,
    hollow_xy: np.ndarray,
    *,
    elements: Union[str, Sequence[str]],
    xy_tol: float = 0.5,
    stacking: Literal["fcc", "hcp"] = "fcc",
    layer_tol: float = 0.5,
    surface_side: Literal["top", "bottom"] = "top",
    reference_layers: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    Classify hollow sites against the first/second subsurface layer on a chosen
    slab side.

    ``hcp`` refers to hollows above the first subsurface layer beneath the exposed
    surface layer. ``fcc`` refers to hollows above the second subsurface layer.
    If the requested subsurface layer is unavailable, an empty array is returned.
    """
    side = str(surface_side).lower()
    if side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'.")
    if reference_layers is None:
        reference_layers = _ordered_layer_indices_by_cluster(
            atoms,
            elements,
            surface_side=side,
            layer_tol=layer_tol,
        )

    ref_layer_idx = 2 if stacking == "fcc" else 1
    if len(reference_layers) <= ref_layer_idx:
        return np.empty((0, 2), dtype=float)
    ref_indices = np.asarray(reference_layers[ref_layer_idx], dtype=int)
    if ref_indices.size == 0:
        return np.empty((0, 2), dtype=float)

    cell = atoms.cell.array
    pbc = atoms.pbc
    ref_xy = atoms.positions[ref_indices, :2]
    selected_hollows = []
    for xy in np.asarray(hollow_xy, dtype=float):
        dxyz = np.zeros((len(ref_xy), 3), dtype=float)
        dxyz[:, :2] = ref_xy - xy[None, :]
        dxyz_mic, _ = find_mic(dxyz, cell, pbc)
        if np.any(np.linalg.norm(dxyz_mic[:, :2], axis=1) < xy_tol):
            selected_hollows.append(_wrap_xy(xy, cell))
    if not selected_hollows:
        return np.empty((0, 2), dtype=float)
    unique = []
    for xy in selected_hollows:
        if not any(np.linalg.norm(xy - other) < 1e-5 for other in unique):
            unique.append(xy)
    return np.asarray(unique, dtype=float)


def _auto_bridge_cutoff(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    scale: float,
) -> float:
    if len(positions) < 2:
        return float("nan")
    nearest = []
    for i in range(len(positions)):
        others = np.delete(positions, i, axis=0)
        disp = np.asarray(others, dtype=float) - np.asarray(positions[i], dtype=float)[None, :]
        disp_mic, _ = find_mic(disp, np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool))
        nearest.append(float(np.min(np.linalg.norm(disp_mic[:, :2], axis=1))))
    return float(np.median(nearest)) * float(scale)


def build_surface_site_registry(
    atoms: Atoms,
    *,
    site_elements: Union[str, Sequence[str]],
    substrate_elements: Optional[Union[str, Sequence[str]]] = None,
    surface_side: Literal["top", "bottom"] = "top",
    site_types: Union[str, Sequence[str]] = ("atop", "bridge", "fcc", "hcp"),
    layer_tol: float = 0.5,
    xy_tol: float = 0.5,
    bridge_cutoff: Optional[float] = None,
    bridge_cutoff_scale: float = 1.15,
    support_xy_tol: float = 2.5,
    termination_site_xy_tol: Optional[float] = None,
    vertical_offset: float = 1.8,
    termination_elements: Sequence[str] = (),
    min_termination_dist: float = 0.75,
) -> List[Dict[str, object]]:
    """
    Build an instantaneous adsorption-site registry for one slab surface.

    Each site record contains:
    - ``site_type``: atop / bridge / fcc / hcp
    - ``xy``: wrapped in-plane position
    - ``anchor_z_A`` and ``suggested_z_A``
    - ``support_indices`` for the metal/support atoms defining the site
    - ``blocked_by_termination`` based on nearby terminations on the same side
    """
    side = str(surface_side).lower()
    if side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'.")
    normalized_site_types = _normalize_site_types(site_types)
    normalized_site_elements = _normalize_symbol_list(site_elements)
    normalized_substrate_elements = (
        _normalize_symbol_list(substrate_elements)
        if substrate_elements is not None
        else ()
    )
    if termination_site_xy_tol is None:
        termination_site_xy_tol = float(support_xy_tol)
    physical_layer_tol = max(float(layer_tol), 0.9) if substrate_elements is not None else float(layer_tol)
    reference_layers = (
        _ordered_layer_indices_by_cluster(
            atoms,
            substrate_elements,
            surface_side=side,
            layer_tol=physical_layer_tol,
        )
        if substrate_elements is not None
        else None
    )
    surface_indices = np.empty((0,), dtype=int)
    if (
        substrate_elements is not None
        and termination_elements
        and set(normalized_site_elements).issubset(set(normalized_substrate_elements))
    ):
        surface_indices = _select_surface_atom_indices_via_terminations(
            atoms,
            normalized_site_elements,
            termination_elements,
            substrate_elements=normalized_substrate_elements,
            surface_side=side,
            layer_tol=layer_tol,
            lateral_tol=termination_site_xy_tol,
        )
    if surface_indices.size == 0:
        surface_indices = _select_surface_atom_indices_by_cluster(
            atoms,
            normalized_site_elements,
            surface_side=side,
            layer_tol=layer_tol,
            reference_elements=substrate_elements,
            reference_layer_tol=physical_layer_tol if substrate_elements is not None else None,
        )
    surface_positions = atoms.positions[surface_indices]
    if len(surface_indices) == 0:
        return []

    def _support_anchor(
        xy: np.ndarray,
        support_indices_for_site: np.ndarray,
    ) -> Tuple[float, float, int]:
        positions = atoms.positions
        slab_midpoint = float(np.mean(positions[:, 2]))
        side_mask = positions[:, 2] >= slab_midpoint if side == "top" else positions[:, 2] <= slab_midpoint
        side_indices = np.flatnonzero(side_mask)
        side_positions = positions[side_indices]
        distances_xy = _xy_distances(xy, side_positions[:, :2], atoms.cell.array, atoms.pbc)
        support_mask = distances_xy <= support_xy_tol
        support_indices_local = side_indices[support_mask]
        support_indices_for_site = np.asarray(support_indices_for_site, dtype=int)
        if support_indices_for_site.size > 0:
            support_indices_local = np.unique(
                np.concatenate((support_indices_local, support_indices_for_site))
            ).astype(int)
        if support_indices_local.size == 0:
            return float("nan"), float("nan"), 0
        support_positions = positions[support_indices_local]
        anchor_z = (
            float(np.max(support_positions[:, 2]))
            if side == "top"
            else float(np.min(support_positions[:, 2]))
        )
        suggested_z = anchor_z + vertical_offset if side == "top" else anchor_z - vertical_offset
        return anchor_z, suggested_z, int(support_indices_local.size)

    def _termination_distance(xy: np.ndarray, suggested_z: float) -> float:
        if not termination_elements or not np.isfinite(suggested_z):
            return float("nan")
        term_mask = np.isin(atoms.get_chemical_symbols(), tuple(termination_elements))
        termination_positions = atoms.positions[term_mask]
        if termination_positions.size == 0:
            return float("nan")
        slab_midpoint = float(np.mean(atoms.positions[:, 2]))
        side_mask = termination_positions[:, 2] >= slab_midpoint if side == "top" else termination_positions[:, 2] <= slab_midpoint
        side_positions = termination_positions[side_mask]
        if side_positions.size == 0:
            return float("nan")
        disp = np.zeros((len(side_positions), 3), dtype=float)
        disp[:, :2] = side_positions[:, :2] - np.asarray(xy, dtype=float)[None, :]
        disp[:, 2] = side_positions[:, 2] - float(suggested_z)
        disp_mic, _ = find_mic(disp, atoms.cell.array, atoms.pbc)
        return float(np.min(np.linalg.norm(disp_mic, axis=1)))

    registry: List[Dict[str, object]] = []
    seen: Dict[str, List[np.ndarray]] = {site_type: [] for site_type in normalized_site_types}
    unique_tol = 1e-4

    def _append_site(site_type: str, xy: np.ndarray, support_indices_for_site: np.ndarray) -> None:
        xy_wrapped = _wrap_xy(np.asarray(xy, dtype=float), atoms.cell.array)
        if any(np.linalg.norm(xy_wrapped - other) < unique_tol for other in seen[site_type]):
            return
        anchor_z, suggested_z, support_atom_count_local = _support_anchor(
            xy_wrapped,
            support_indices_for_site,
        )
        termination_dist = _termination_distance(xy_wrapped, suggested_z)
        registry.append(
            {
                "site_type": site_type,
                "surface_side": side,
                "xy": xy_wrapped,
                "support_indices": np.asarray(support_indices_for_site, dtype=int),
                "anchor_z_A": anchor_z,
                "suggested_z_A": suggested_z,
                "support_atom_count_local": int(support_atom_count_local),
                "nearest_termination_dist_A": termination_dist,
                "blocked_by_termination": bool(
                    np.isfinite(termination_dist) and termination_dist < float(min_termination_dist)
                ),
            }
        )
        seen[site_type].append(xy_wrapped)

    if "atop" in normalized_site_types:
        for support_idx in surface_indices:
            _append_site("atop", atoms.positions[int(support_idx), :2], np.asarray([support_idx], dtype=int))

    if "bridge" in normalized_site_types and len(surface_indices) >= 2:
        cutoff = float(bridge_cutoff) if bridge_cutoff is not None else _auto_bridge_cutoff(
            surface_positions,
            atoms.cell.array,
            atoms.pbc,
            bridge_cutoff_scale,
        )
        if np.isfinite(cutoff):
            for i_local, idx_i in enumerate(surface_indices[:-1]):
                for idx_j in surface_indices[i_local + 1 :]:
                    vector = np.asarray(
                        atoms.get_distances(int(idx_i), int(idx_j), mic=True, vector=True),
                        dtype=float,
                    )
                    if float(np.linalg.norm(vector[:2])) <= cutoff:
                        xy = _midpoint_xy(
                            atoms.positions[int(idx_i)],
                            atoms.positions[int(idx_j)],
                            atoms.cell.array,
                            atoms.pbc,
                        )
                        _append_site("bridge", xy, np.asarray([idx_i, idx_j], dtype=int))

    if any(site in normalized_site_types for site in ("fcc", "hcp")) and len(surface_indices) >= 3:
        hollow_xy = get_hollow_xy(surface_positions[:, :2], atoms.cell.array)
        hollow_elements = substrate_elements if substrate_elements is not None else site_elements
        hollow_match_tol = float(xy_tol)
        if substrate_elements is not None:
            # Rough MXene surfaces can shift the projected subsurface registry by more
            # than the strict site-matching tolerance used for occupancy bookkeeping.
            hollow_match_tol = max(hollow_match_tol, 0.8)
        for site_type in ("fcc", "hcp"):
            if site_type not in normalized_site_types:
                continue
            selected_xy = classify_hollow_sites_on_surface(
                atoms,
                hollow_xy,
                elements=hollow_elements,
                xy_tol=hollow_match_tol,
                stacking=site_type,
                layer_tol=physical_layer_tol,
                surface_side=side,
                reference_layers=reference_layers,
            )
            for xy in np.asarray(selected_xy, dtype=float):
                distances_xy = _xy_distances(
                    xy,
                    surface_positions[:, :2],
                    atoms.cell.array,
                    atoms.pbc,
                )
                order = np.argsort(distances_xy)
                support = surface_indices[order[: min(3, len(order))]]
                _append_site(site_type, xy, np.asarray(support, dtype=int))

    return registry


def classify_hollow_sites(
    atoms: Atoms,
    hollow_xy: np.ndarray,
    element: str = "Ti",
    xy_tol: float = 0.5,
    stacking: Literal["fcc", "hcp"] = "fcc",
    layer_tol: float = 0.5,
) -> np.ndarray:
    """
    Select hollow sites above a specific substrate layer (fcc = third, hcp = second).

    Args:
        atoms: ASE Atoms object.
        hollow_xy: Candidate hollow site xy positions ((M,2) array).
        element: Substrate atom symbol.
        xy_tol: Lateral tolerance for matching hollow to substrate site.
        stacking: "fcc" or "hcp".
        layer_tol: Tolerance for clustering substrate z-layers.

    Returns:
        (K,2) array of selected hollow xy positions.
    """
    subs = [atom for atom in atoms if atom.symbol == element]
    if not subs:
        raise ValueError(f"No atoms with symbol '{element}' in atoms object.")
    zvals = np.array([atom.position[2] for atom in subs])
    layer_clusters = _cluster_axis_values(zvals, tol=layer_tol)
    layer_clusters = sorted(
        layer_clusters,
        key=lambda cluster: float(np.mean(zvals[cluster])),
        reverse=True,
    )
    if len(layer_clusters) < 3:
        raise RuntimeError("Not enough substrate layers to classify hollows.")
    idx = 2 if stacking == "fcc" else 1
    ref_atoms = [subs[int(i)] for i in layer_clusters[idx]]

    # Prepare the cell for PBC operations.
    cell = atoms.cell.array  # Shape (3, 3).
    pbc = atoms.pbc

    selected_hollows = []
    for xy in hollow_xy:
        for atom in ref_atoms:
            # Create a 3D displacement vector (z = 0).
            dxyz = np.zeros(3)
            dxyz[:2] = atom.position[:2] - xy
            # Use find_mic for proper PBC handling.
            dxyz_mic, _ = find_mic(dxyz.reshape(1, 3), cell, pbc)
            if np.linalg.norm(dxyz_mic[0][:2]) < xy_tol:
                selected_hollows.append(xy)
                break
    return np.array(selected_hollows)


def generate_adsorbate_configuration(
    atoms: Atoms,
    site_type: Literal["atop", "fcc", "hcp"] = "fcc",
    element: str = "Cu",
    coverage: float = 1.0,
    xy_tol: float = 0.5,
    support_xy_tol: float = 2.5,
    vertical_offset: float = 1.8,
    substrate_element: str = "Ti",
    seed: int = 42,
) -> Atoms:
    """
    Place adsorbates at specified sites using padded hollow-site logic and
    always place the adsorbate above the highest *any* atom within support_xy_tol.
    Handles multilayer coverages (e.g., 2.5 ML).

    Args:
        atoms: ASE Atoms object (substrate, possibly with functionals).
        site_type: 'atop', 'fcc', or 'hcp'.
        element: Adsorbate symbol (e.g. 'Cu').
        coverage: Fractional ML to fill (e.g., 2.0 = 2 layers).
        xy_tol: Tolerance for hollow site classification.
        support_xy_tol: Lateral tolerance for finding the highest supporting atom.
        vertical_offset: Height above the highest local atom for placement.
        substrate_element: Substrate atom symbol (e.g. 'Ti').
        seed: RNG seed.

    Returns:
        New Atoms object (deep copy) with adsorbates added.
    """
    rng = np.random.default_rng(seed)
    atoms_new = atoms.copy()
    cell = atoms.get_cell()
    # Site registry.
    if site_type == "atop":
        site_xy = get_toplayer_xy(atoms, element=substrate_element)
    else:
        atop_xy = get_toplayer_xy(atoms, element=substrate_element)
        hollows = get_hollow_xy(atop_xy, cell)
        site_xy = classify_hollow_sites(
            atoms, hollows, element=substrate_element, xy_tol=xy_tol, stacking=site_type
        )
    n_sites = len(site_xy)
    if n_sites == 0:
        raise RuntimeError("*** NO REGISTRY SITES AVAILABLE ***")
    for layer_indices in _select_site_layers_for_coverage(n_sites, coverage, rng):
        for xy in site_xy[np.asarray(layer_indices, dtype=int)]:
            neighbors_z = []
            for atom in atoms_new:
                dxyz = np.zeros(3)
                dxyz[:2] = atom.position[:2] - xy
                dxyz_mic, _ = find_mic(dxyz.reshape(1, 3), cell, atoms_new.pbc)
                if np.linalg.norm(dxyz_mic[0][:2]) < support_xy_tol:
                    neighbors_z.append(atom.position[2])
            if not neighbors_z:
                raise RuntimeError(
                    f"No support found at site {xy} for adsorbate placement."
                )
            z_max = max(neighbors_z)
            pos = np.array([xy[0], xy[1], z_max + vertical_offset])
            atoms_new.append(Atom(element, pos))
    return atoms_new


def _load_adsorbate_template(
    adsorbate: Union[str, Atoms],
) -> Atoms:
    """Load a monoatomic or molecular adsorbate template."""
    if isinstance(adsorbate, Atoms):
        return adsorbate.copy()
    if isinstance(adsorbate, str):
        if os.path.exists(adsorbate):
            return read(adsorbate)
        symbols = string2symbols(adsorbate)
        if len(symbols) != 1:
            raise ValueError(
                "String adsorbates must be a single chemical symbol or a structure file."
            )
        return Atoms(symbols=symbols, positions=[(0.0, 0.0, 0.0)])
    raise TypeError("adsorbate must be an ASE Atoms object or a string.")


def place_adsorbate_on_site(
    atoms: Atoms,
    adsorbate: Union[str, Atoms],
    site: Dict[str, object],
    *,
    anchor_index: int = 0,
) -> Tuple[Atoms, np.ndarray]:
    """
    Place one adsorbate on a specific site-registry entry.

    Args:
        atoms: Base slab structure.
        adsorbate: Monoatomic symbol or ASE adsorbate template.
        site: One row from ``build_surface_site_registry(...)``.
        anchor_index: Anchor atom index within the adsorbate template.

    Returns:
        Tuple of:
          - new structure with the adsorbate placed
          - support/site-defining slab indices for the chosen site
    """
    adsorbate_template = _load_adsorbate_template(adsorbate)
    if not (0 <= int(anchor_index) < len(adsorbate_template)):
        raise ValueError("anchor_index is out of range for the adsorbate template.")

    suggested_z = float(site.get("suggested_z_A", np.nan))
    if not np.isfinite(suggested_z):
        raise ValueError("Selected site does not define a finite adsorption height.")

    atoms_new = atoms.copy()
    anchor_pos = np.array(
        [
            float(site["xy"][0]),
            float(site["xy"][1]),
            suggested_z,
        ],
        dtype=float,
    )
    relative = (
        adsorbate_template.get_positions()
        - adsorbate_template.positions[int(anchor_index)].copy()
    )
    for symbol, rel in zip(adsorbate_template.get_chemical_symbols(), relative):
        atoms_new.append(Atom(symbol, anchor_pos + rel))

    return atoms_new, np.asarray(site["support_indices"], dtype=int)


def select_surface_atom_indices(
    atoms: Atoms,
    elements: Union[str, Sequence[str]],
    *,
    surface_side: Literal["top", "bottom"] = "top",
    split_method: Literal["count"] = "count",
    layer_tol: float = 0.5,
) -> np.ndarray:
    """
    Select one exposed surface layer from a two-sided slab.

    ``split_method='count'`` is retained for backward compatibility. The
    implementation uses tolerance-based z clustering so it also works for
    multicomponent surfaces where one surface layer spans several elements.
    """
    if split_method != "count":
        raise ValueError("Only split_method='count' is currently supported.")
    return np.sort(
        _select_surface_atom_indices_by_cluster(
            atoms,
            elements,
            surface_side=surface_side,
            layer_tol=layer_tol,
        ).astype(int)
    )


def initialize_surface_adsorbates(
    atoms: Atoms,
    adsorbate: Union[str, Atoms],
    *,
    n_adsorbates: int,
    support_element: Optional[str] = None,
    site_elements: Optional[Union[str, Sequence[str]]] = None,
    substrate_elements: Optional[Union[str, Sequence[str]]] = None,
    surface_side: Literal["top", "bottom"] = "top",
    site_types: Union[str, Sequence[str]] = "atop",
    layer_tol: float = 0.5,
    xy_tol: float = 0.5,
    bridge_cutoff: Optional[float] = None,
    bridge_cutoff_scale: float = 1.15,
    support_xy_tol: float = 2.5,
    termination_site_xy_tol: Optional[float] = None,
    vertical_offset: float = 1.8,
    termination_elements: Sequence[str] = (),
    min_termination_dist: float = 0.75,
    anchor_index: int = 0,
    seed: int = 42,
) -> Tuple[Atoms, np.ndarray, np.ndarray]:
    """
    Place a fixed number of adsorbates on a chosen slab surface using an
    instantaneous high-symmetry site registry.

    Returns:
        atoms_new: structure with adsorbates added
        selected_support_indices: chosen support/site-defining atom indices
        candidate_support_indices: all eligible support/site-defining atom indices
    """
    adsorbate_template = _load_adsorbate_template(adsorbate)
    if not (0 <= int(anchor_index) < len(adsorbate_template)):
        raise ValueError("anchor_index is out of range for the adsorbate template.")
    if int(n_adsorbates) < 1:
        raise ValueError("n_adsorbates must be >= 1.")

    if site_elements is None:
        if support_element is None:
            raise ValueError("Provide either support_element or site_elements.")
        site_elements = support_element

    registry = build_surface_site_registry(
        atoms,
        site_elements=site_elements,
        substrate_elements=substrate_elements,
        surface_side=surface_side,
        site_types=site_types,
        layer_tol=layer_tol,
        xy_tol=xy_tol,
        bridge_cutoff=bridge_cutoff,
        bridge_cutoff_scale=bridge_cutoff_scale,
        support_xy_tol=support_xy_tol,
        termination_site_xy_tol=termination_site_xy_tol,
        vertical_offset=vertical_offset,
        termination_elements=termination_elements,
        min_termination_dist=min_termination_dist,
    )
    candidate_sites = [
        row
        for row in registry
        if np.isfinite(float(row["suggested_z_A"]))
        and not bool(row["blocked_by_termination"])
    ]
    if not candidate_sites:
        raise RuntimeError("No eligible adsorption sites were generated.")
    if int(n_adsorbates) > len(candidate_sites):
        raise ValueError(
            f"n_adsorbates must satisfy 1 <= n_adsorbates <= {len(candidate_sites)}."
        )

    rng = np.random.default_rng(seed)
    selected_site_indices = np.asarray(
        rng.choice(len(candidate_sites), size=int(n_adsorbates), replace=False),
        dtype=int,
    )
    selected_sites = [candidate_sites[int(i)] for i in selected_site_indices]

    selected_support_groups = []
    atoms_new = atoms.copy()
    for site in selected_sites:
        atoms_new, support_indices = place_adsorbate_on_site(
            atoms_new,
            adsorbate_template,
            site,
            anchor_index=anchor_index,
        )
        selected_support_groups.append(np.asarray(support_indices, dtype=int))

    selected_support_indices = (
        np.unique(np.concatenate(selected_support_groups))
        if selected_support_groups
        else np.array([], dtype=int)
    )
    candidate_support_indices = np.unique(
        np.concatenate(
            [
                np.asarray(site["support_indices"], dtype=int)
                for site in candidate_sites
            ]
        )
    )

    return atoms_new, selected_support_indices, candidate_support_indices


def initialize_alloy_sublattice(
    atoms: Atoms, site_element: str, composition: dict, seed: int = 67
) -> Atoms:
    """
    Replaces atoms of type `site_element` with a random mixture.
    """
    rng = np.random.default_rng(seed)
    new_atoms = atoms.copy()

    site_indices = [
        i for i, s in enumerate(new_atoms.get_chemical_symbols()) if s == site_element
    ]
    n_sites = len(site_indices)

    if n_sites == 0:
        raise ValueError(f"No atoms of type '{site_element}' found.")

    counts = {}
    current_count = 0
    elements = list(composition.keys())

    for el in elements[:-1]:
        n_el = int(round(composition[el] * n_sites))
        counts[el] = n_el
        current_count += n_el
    counts[elements[-1]] = n_sites - current_count

    sublattice_symbols = []
    for el, count in counts.items():
        sublattice_symbols.extend([el] * count)

    rng.shuffle(sublattice_symbols)

    all_symbols = np.array(new_atoms.get_chemical_symbols())
    all_symbols[site_indices] = sublattice_symbols
    new_atoms.set_chemical_symbols(all_symbols)

    return new_atoms
