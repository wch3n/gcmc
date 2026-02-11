"""
utils.py

Utility functions for MC simulation initial configuration generation on periodic slabs,
including robust, PBC-aware hollow-site registry (via padding) and multilayer adsorbate
placement (works for any coverage, e.g., 2.5 ML).

All functions compatible with ASE Atoms objects.
"""

import numpy as np
from typing import List, Literal, Optional, Sequence
from ase import Atom, Atoms
from ase.geometry import find_mic
from scipy.spatial import Delaunay


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


def classify_hollow_sites(
    atoms: Atoms,
    hollow_xy: np.ndarray,
    element: str = "Ti",
    xy_tol: float = 0.5,
    stacking: Literal["fcc", "hcp"] = "fcc",
) -> np.ndarray:
    """
    Select hollow sites above a specific substrate layer (fcc = third, hcp = second).

    Args:
        atoms: ASE Atoms object.
        hollow_xy: Candidate hollow site xy positions ((M,2) array).
        element: Substrate atom symbol.
        xy_tol: Lateral tolerance for matching hollow to substrate site.
        stacking: "fcc" or "hcp".

    Returns:
        (K,2) array of selected hollow xy positions.
    """
    subs = [atom for atom in atoms if atom.symbol == element]
    zvals = np.array([atom.position[2] for atom in subs])
    unique_layers = np.unique(np.round(zvals, 0))
    z_layers = np.sort(unique_layers)[::-1]
    if len(z_layers) < 3:
        raise RuntimeError("Not enough substrate layers to classify hollows.")
    idx = 2 if stacking == "fcc" else 1
    ref_z = z_layers[idx]
    ref_atoms = [
        atom for atom in subs if abs(np.round(atom.position[2], 0) - ref_z) < 1e-3
    ]

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
    # Handle multilayer coverage.
    n_layers = int(np.floor(coverage))
    frac_layer = coverage - n_layers
    # Add an integer number of layers.
    for ilayer in range(n_layers):
        for xy in site_xy:
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
    # Add a partial layer if needed.
    if frac_layer > 1e-8:
        n_partial = int(round(frac_layer * n_sites))
        chosen = rng.choice(n_sites, n_partial, replace=False)
        for xy in site_xy[chosen]:
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
