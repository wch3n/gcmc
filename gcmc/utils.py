"""
utils.py

Utility functions for MC simulation initial configuration generation on periodic slabs,
including robust, PBC-aware hollow-site registry (via padding) and multilayer adsorbate
placement (works for any coverage, e.g., 2.5 ML).

All functions compatible with ASE Atoms objects.
"""

import numpy as np
from typing import List, Tuple, Literal, Optional
from ase import Atom, Atoms
from ase.geometry import find_mic
from scipy.spatial import Delaunay

def get_toplayer_xy(
    atoms: Atoms,
    element: str = "Ti",
    z_tol: float = 0.3
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
    xy = np.array([
        atom.position[:2]
        for atom in atoms_elem
        if atom.position[2] > z_max - z_tol
    ])
    return xy

def get_hollow_xy(
    toplayer_xy: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Find hollow sites for a periodic slab by padding the cell with image points.

    Args:
        toplayer_xy: (N,2) array of top-layer xy positions.
        cell: (3,3) simulation cell.

    Returns:
        (M,2) array of unique hollow xy positions within the center cell.
    """
    a1, a2 = cell[0, :2], cell[1, :2]
    # Tile ±1 in x and y (total 9 images)
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
        # Check if center is inside the original cell (using fractional coordinates)
        cell_2d = np.array([a1, a2]).T
        try:
            frac = np.linalg.solve(cell_2d, center)
        except np.linalg.LinAlgError:
            continue
        if np.all((frac >= -1e-8) & (frac < 1 - 1e-8)):
            hollow_xy.append(center)
    # Remove duplicates
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
    stacking: Literal["fcc", "hcp"] = "fcc"
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
    idx = 2 if stacking == 'fcc' else 1
    ref_z = z_layers[idx]
    ref_atoms = [atom for atom in subs if abs(np.round(atom.position[2],0) - ref_z) < 1e-3]

    # Prepare cell for PBC
    cell = atoms.cell.array  # This is (3,3)
    pbc = atoms.pbc

    selected_hollows = []
    for xy in hollow_xy:
        for atom in ref_atoms:
            # Create a 3D displacement vector (z=0)
            dxyz = np.zeros(3)
            dxyz[:2] = atom.position[:2] - xy
            # Use find_mic for proper PBC
            dxyz_mic, _ = find_mic(dxyz.reshape(1,3), cell, pbc)
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
    # Site registry
    if site_type == "atop":
        site_xy = get_toplayer_xy(atoms, element=substrate_element)
    else:
        atop_xy = get_toplayer_xy(atoms, element=substrate_element)
        hollows = get_hollow_xy(atop_xy, cell)
        site_xy = classify_hollow_sites(
            atoms, hollows, element=substrate_element, xy_tol=xy_tol, stacking=site_type
        )
    n_sites = len(site_xy)
    print(n_sites)
    if n_sites == 0:
        raise RuntimeError("*** NO REGISTRY SITES AVAILABLE ***")
    # Handle multilayer coverage
    n_layers = int(np.floor(coverage))
    frac_layer = coverage - n_layers
    # Add integer number of layers
    for ilayer in range(n_layers):
        for xy in site_xy:
            neighbors_z = [
                atom.position[2]
                for atom in atoms_new
                if np.linalg.norm(atom.position[:2] - xy) < support_xy_tol
            ]
            if not neighbors_z:
                raise RuntimeError(f"No support found at site {xy} for adsorbate placement.")
            z_max = max(neighbors_z)
            pos = np.array([xy[0], xy[1], z_max + vertical_offset])
            atoms_new.append(Atom(element, pos))
    # Add partial layer if needed
    if frac_layer > 1e-8:
        n_partial = int(round(frac_layer * n_sites))
        chosen = rng.choice(n_sites, n_partial, replace=False)
        for xy in site_xy[chosen]:
            neighbors_z = [
                atom.position[2]
                for atom in atoms_new
                if np.linalg.norm(atom.position[:2] - xy) < support_xy_tol
            ]
            if not neighbors_z:
                raise RuntimeError(f"No support found at site {xy} for adsorbate placement.")
            z_max = max(neighbors_z)
            pos = np.array([xy[0], xy[1], z_max + vertical_offset])
            atoms_new.append(Atom(element, pos))
    return atoms_new
